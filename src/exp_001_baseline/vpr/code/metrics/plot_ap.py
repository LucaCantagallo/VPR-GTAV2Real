import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

    results_dir = os.path.join(work_dir, "ap_results")
    plot_dir = os.path.join(results_dir, "plots_ap")
    os.makedirs(plot_dir, exist_ok=True)

    csv_files = glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"Nessun file AP trovato in {results_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        name = os.path.splitext(os.path.basename(csv_path))[0]

        # Estrai metriche AP@K
        ap_cols = [c for c in df.columns if c.startswith("AP@")]
        ap_values = df[ap_cols].iloc[0].values
        ks = [int(c.split("@")[1]) for c in ap_cols]

        # Plot barre AP@K
        plt.figure(figsize=(6, 4))
        plt.bar(ks, ap_values, width=3)
        plt.title(f"{name} - AP@K")
        plt.xlabel("K")
        plt.ylabel("AP@K")
        plt.xticks(ks)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{name}_ap_at_k.png"))
        plt.close()

        # Stampa Mean AP
        mean_ap = df["Mean_AP"].iloc[0]
        print(f"{name}: Mean AP = {mean_ap:.4f}")
