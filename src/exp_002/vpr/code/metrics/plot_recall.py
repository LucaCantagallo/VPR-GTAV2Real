import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_params, get_n_folders

if __name__ == "__main__":
    experiments_dir = "./experiments"
    config_file = "./recall.yaml"
    params = load_params(config_file)

    base_path = os.path.join(experiments_dir, params["dataset"])
    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    elif str(params["work_dir"]) == "-1":
        subfolders = sorted(
            [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        )
        if not subfolders:
            raise ValueError(f"Nessuna sottocartella trovata in {base_path}")
        work_dir = os.path.join(base_path, subfolders[-1])
    else:
        work_dir = os.path.join(base_path, str(params["work_dir"]))

    # Cartella dove salvare i grafici
    plot_dir = os.path.join(work_dir, "recall_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Trova tutti i file recall_*.txt
    recall_files = glob(os.path.join(work_dir, "recall_*.txt"))
    if not recall_files:
        print(f"Nessun file recall_*.txt trovato in {work_dir}")
        exit()

    # Plot per ogni file
    for path in recall_files:
        df = pd.read_csv(path, sep=" ", header=None)
        df.index += 1
        name = os.path.basename(path).replace(".txt", "")
        
        plt.figure()
        plt.plot(df.index, df[0], marker="o", label=name)
        plt.xlabel("K")
        plt.ylabel("Recall")
        plt.title(f"Recall@K - {name}")
        plt.grid(True)
        plt.legend()
        
        out_path = os.path.join(plot_dir, f"{name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Grafico salvato in {out_path}")

    # Plot cumulativo (tutti insieme)
    plt.figure()
    for path in recall_files:
        df = pd.read_csv(path, sep=" ", header=None)
        df.index += 1
        name = os.path.basename(path).replace(".txt", "")
        plt.plot(df.index, df[0], marker="o", label=name)

    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.title("Recall@K Comparison")
    plt.grid(True)
    plt.legend()
    combined_path = os.path.join(plot_dir, "recall_comparison.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"Grafico cumulativo salvato in {combined_path}")
