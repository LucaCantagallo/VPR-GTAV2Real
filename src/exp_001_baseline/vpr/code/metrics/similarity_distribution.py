import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_n_folders, load_params


if __name__ == "__main__":
    experiments_dir = "./experiments"
    config_file = "./recall.yaml"
    params = load_params(config_file)

    base_path = os.path.join(experiments_dir, params["dataset"])
    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    elif str(params["work_dir"]) == "-1":
        # Prende l'ultimo elemento numerico dentro base_path (ordinamento numerico corretto)
        subfolders = [
            f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()
        ]
        if not subfolders:
            raise ValueError(f"Nessuna sottocartella trovata in {base_path}")
        
        # Ordina numericamente invece che alfabeticamente
        subfolders = sorted(subfolders, key=lambda x: int(x))
        work_dir = os.path.join(base_path, subfolders[-1])
    else:
        work_dir = os.path.join(base_path, str(params["work_dir"]))

    matrix_paths = glob(os.path.join(work_dir, "cm_*.txt"))
    plot_dir = os.path.join(work_dir, "plots_similarity")
    os.makedirs(plot_dir, exist_ok=True)

    for path in matrix_paths:
        matrix = np.loadtxt(path)
        name = os.path.splitext(os.path.basename(path))[0]

        # Estrai valori same-place (diagonale) e different-place (off-diagonale)
        same_place = np.diag(matrix)
        different_place = matrix[~np.eye(matrix.shape[0], dtype=bool)]

        # Grafico distribuzione
        plt.figure(figsize=(7, 4))
        sns.kdeplot(same_place, label="Same place", fill=True)
        sns.kdeplot(different_place, label="Different place", fill=True)
        plt.title(f"Distribuzione Similarità Coseno - {name}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{name}_similarity_distribution.png"))
        plt.close()

        # Stampa statistiche base
        print(f"\n{name}")
        print(f"Same-place mean: {np.mean(same_place):.4f}")
        print(f"Different-place mean: {np.mean(different_place):.4f}")
        print(f"Gap medio: {np.mean(same_place) - np.mean(different_place):.4f}")
