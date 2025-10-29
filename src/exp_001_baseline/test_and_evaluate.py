import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from datetime import datetime
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import TestDataset
from models import MLPCosine
from utils import load_params, get_n_folders

def compute_cosine_matrix(features0, features1):
    cm = np.zeros((len(features0), len(features1)))
    for i in range(len(features0)):
        cm[i] = F.cosine_similarity(features0[i].unsqueeze(0), features1).cpu().numpy()
    return cm

def compute_top_k_accuracy(matrix, k_list):
    res = np.argsort(matrix, axis=1)
    accuracies = {}
    for k in k_list:
        correct = 0
        for i in range(len(matrix)):
            top_k = res[i, -k:]
            if i in top_k:
                correct += 1
        accuracies[k] = correct / len(matrix)
    return accuracies

if __name__ == "__main__":
    # --- Load config ---
    config_file = "./test_and_evaluate.yaml"
    params = load_params(config_file)
    dataset_name = params["dataset"]
    k_eval = params["k"]

    experiments_dir = "./experiments"
    base_path = os.path.join(experiments_dir, dataset_name)
    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    else:
        work_dir = os.path.join(base_path, params["work_dir"])

    # --- Create / reset folders ---
    test_dir = os.path.join(work_dir, "test")
    recall_dir = os.path.join(work_dir, "recall")
    k_top_dir = os.path.join(work_dir, "k_top")
    for d in [test_dir, recall_dir, k_top_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # --- Dataset ---
    root = "/home/lcantagallo/VPR-GTAV2Real/dataset/Tokyo247/Tokyo_24_7"
    places = glob(os.path.join(root, "*"))
    places = [places[i] for i in range(len(places)) if i % 3 == 0 or i % 3 == 2]
    dataset = TestDataset(places)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

    places_names = [os.path.split(p)[-1].split(".")[0] for p in places]

    # --- Load model (best only) ---
    model_paths = sorted(glob(os.path.join(work_dir, "*best_loss*.pt")))
    if not model_paths:
        raise ValueError("No best_loss model found in work_dir")
    model_path = model_paths[0]
    model = MLPCosine()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # --- Extract features ---
    features = []
    for mb in tqdm(dataloader, desc="Extracting features"):
        f = model(mb.to(device))
        features.append(f.detach().cpu())
    features = torch.cat(features)

    day_features = features[::2]
    night_features = features[1::2]

    # --- Compute cosine matrices ---
    cm_day = compute_cosine_matrix(day_features, night_features)
    cm_night = compute_cosine_matrix(night_features, day_features)

    # Save matrices
    np.savetxt(os.path.join(test_dir, "cm_day_best_loss.txt"), cm_day)
    np.savetxt(os.path.join(test_dir, "cm_night_best_loss.txt"), cm_night)

    # --- Replicate old recall.py behavior ---
    for cm, name_suffix in zip([cm_day, cm_night], ["day", "night"]):
        for path_name, matrix in zip([f"cm_{name_suffix}_best_loss.txt"], [cm]):
            res = np.argsort(matrix, axis=1)
            recall_list = []
            for j in range(1, k_eval+1):
                accuracy = 0
                for i in range(len(matrix)):
                    top_k = res[i, -j:]
                    if i in top_k:
                        accuracy += 1
                accuracy /= len(matrix)
                recall_list.append(accuracy)
            # Save detailed recall
            df = pd.DataFrame(recall_list)
            df.index += 1
            df.to_csv(os.path.join(recall_dir, f"recall_{name_suffix}.txt"), sep=" ", header=None)

    # --- Replicate old top_k.py behavior ---
    for cm, name_suffix in zip([cm_day, cm_night], ["day", "night"]):
        res = np.argsort(cm, axis=1)
        for i in range(len(cm)):
            top_k = res[i, -k_eval:]
            to_print = f"{places_names[i]}: " + " ".join([places_names[e] for e in top_k]) + "\n"
            # Append to file
            with open(os.path.join(k_top_dir, f"top_{k_eval}_{name_suffix}.txt"), "a") as f:
                accuracy = sum([1 if i in res[i, -k_eval:] else 0 for i in range(len(cm))]) / len(cm)
                f.write(str(accuracy) + "\n" + to_print)

    # --- Final evaluation summary ---
    k_list = [1,5,10,k_eval]
    recall_day = compute_top_k_accuracy(cm_day, k_list)
    recall_night = compute_top_k_accuracy(cm_night, k_list)

    summary_path = os.path.join(work_dir, "final_evaluation.txt")
    train_config_path = os.path.join(work_dir, "train.yaml")


    with open(summary_path, "w") as f:
        f.write(f"Final Evaluation Summary\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"k: {k_eval}\n")
        f.write(f"Number of models evaluated: 1\n")
        f.write(f"Number of query images evaluated: {cm_day.shape[0]} (each day/night)\n")
        for k in k_list:
            f.write(f"Top-{k} recall (day/night/avg): {recall_day[k]:.4f} / {recall_night[k]:.4f} / {(recall_day[k]+recall_night[k])/2:.4f}\n")
        f.write(f"Generated on: {datetime.now()}\n")
        f.write("\n\n")
        f.write("Information about the training process:\n")
        if os.path.exists(train_config_path):
            with open(train_config_path, "r") as train_file:
                f.write(train_file.read())
        else:
            f.write("train.yaml not found.\n")


    print(f"Final evaluation summary saved to {summary_path}")
