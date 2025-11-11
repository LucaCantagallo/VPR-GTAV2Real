import numpy as np
import os
from glob import glob
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
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
    k_values = [1, 5, 10]

    results_dir = os.path.join(work_dir, "ap_results")
    os.makedirs(results_dir, exist_ok=True)

    for path in matrix_paths:
        matrix = np.loadtxt(path)
        name = os.path.splitext(os.path.basename(path))[0]

        gt = np.arange(len(matrix))
        y_true = np.zeros_like(matrix)
        y_true[np.arange(len(matrix)), gt] = 1  # 1 per il match corretto

        ap_scores = []
        ap_at_k = {k: [] for k in k_values}

        for i in range(len(matrix)):
            y_score = matrix[i]
            y_true_i = y_true[i]

            # Average Precision (su tutta la classifica)
            ap = average_precision_score(y_true_i, y_score)
            ap_scores.append(ap)

            # AP@K
            sorted_idx = np.argsort(y_score)[::-1]
            for k in k_values:
                top_k = sorted_idx[:k]
                relevant_in_topk = y_true_i[top_k].sum()
                ap_at_k[k].append(relevant_in_topk / k)

        mean_ap = np.mean(ap_scores)
        mean_ap_at_k = {k: np.mean(v) for k, v in ap_at_k.items()}

        print(f"\n{name}")
        print(f"Mean AP: {mean_ap:.4f}")
        for k in k_values:
            print(f"AP@{k}: {mean_ap_at_k[k]:.4f}")

        # Salva risultati
        df = pd.DataFrame({
            "Mean_AP": [mean_ap],
            **{f"AP@{k}": [mean_ap_at_k[k]] for k in k_values}
        })
        df.to_csv(os.path.join(results_dir, f"{name}_ap.csv"), index=False)
