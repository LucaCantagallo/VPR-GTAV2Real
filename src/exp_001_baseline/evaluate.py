import os
import numpy as np
import pandas as pd
from glob import glob
import yaml
from datetime import datetime

# --- Utility ---
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Funzioni per top-k e recall ---
def compute_top_k(cm, places_names, k):
    accuracy = 0
    res = np.argsort(cm, axis=1)
    for i in range(len(cm)):
        top_k_idx = res[i, -k:]
        if i in top_k_idx:
            accuracy += 1
    accuracy /= len(cm)
    return accuracy

def compute_recall(cm, k_list=[1,5,10,50]):
    res = np.argsort(cm, axis=1)
    recalls = {}
    for k in k_list:
        accuracy = 0
        for i in range(len(cm)):
            top_k_idx = res[i, -k:]
            if i in top_k_idx:
                accuracy += 1
        recalls[k] = accuracy / len(cm)
    return recalls

# --- Main Evaluation ---
if __name__ == "__main__":
    # --- Config ---
    config_file = "./test_and_evaluate.yaml"
    with open(config_file, "r") as f:
        params = yaml.safe_load(f)
    
    dataset = params.get("dataset", "gta")
    work_dir = params.get("work_dir", None)
    k = params.get("k", 50)
    
    experiments_dir = "./experiments"
    base_path = os.path.join(experiments_dir, dataset)
    
    if work_dir is None:
        n_folders = len([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,d))])
        work_dir = os.path.join(base_path, str(n_folders))
    else:
        work_dir = os.path.join(base_path, str(work_dir))
    
    # --- Creazione cartelle per intermedi ---
    recall_dir = os.path.join(work_dir, "recall")
    topk_dir = os.path.join(work_dir, "k_top")
    mkdir_if_not_exist(recall_dir)
    mkdir_if_not_exist(topk_dir)
    
    # --- Trova le matrici cm generate da test.py ---
    cm_paths = glob(os.path.join(work_dir, "cm_*.txt"))
    all_accuracies = []
    all_recalls = []
    
    for cm_path in cm_paths:
        name = os.path.split(cm_path)[-1].replace(".txt","")
        cm = np.loadtxt(cm_path)
        
        # --- Top-k ---
        top_k_acc = compute_top_k(cm, range(len(cm)), k)
        with open(os.path.join(topk_dir, f"top_{k}_{name}.txt"), "w") as f:
            f.write(f"Top-{k} accuracy: {top_k_acc}\n")
        all_accuracies.append(top_k_acc)
        
        # --- Recall ---
        recalls = compute_recall(cm, k_list=[1,5,10,50])
        recall_file = os.path.join(recall_dir, f"recall_{name}.txt")
        df = pd.DataFrame([recalls])
        df.to_csv(recall_file, sep=" ", index=False)
        all_recalls.append(recalls)
    
    # --- File finale sintetico ---
    mean_acc = np.mean(all_accuracies)
    
    final_file = os.path.join(work_dir, "final_evaluation.txt")
    with open(final_file, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"k: {k}\n")
        f.write(f"Top-{k} mean accuracy (best_loss): {mean_acc:.4f}\n\n")
        f.write("Recall (media tra tutte le matrici):\n")
        for key in [1,5,10,50]:
            mean_recall = np.mean([r[key] for r in all_recalls])
            f.write(f"Recall@{key}: {mean_recall:.4f}\n")
        f.write(f"\nEvaluation completed at: {datetime.now()}\n")
    
    print(f"Evaluation done. Summary saved at {final_file}")
