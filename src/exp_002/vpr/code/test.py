## test.py

import yaml
import os
from glob import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import TestDataset
from models import MLPCosine
from utils import load_params, get_n_folders

def compute_cm(features0, features1, cm, j, name):
    for i in range(len(features0)):
        cosine_sim = F.cosine_similarity(features0[i].unsqueeze(0), features1)
        cm[i] = cosine_sim
    np.savetxt(os.path.join(work_dir, f"cm_{name}_{model_names[j]}.txt"), cm)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    experiments_dir = "./experiments"
    config_file = "./pipeline.yaml"
    params = load_params(config_file)

    if params["dataload"] == "daynight":
        from data_loader_daynight import load_dataset
    elif params["dataload"] == "vpr":
        from data_loader_vpr import load_dataset
    else:
        raise ValueError(f"Dataload {params['dataload']} non supportato")


    base_path = os.path.join(experiments_dir, params["save_dir"])

    # Determina la cartella di lavoro
    if params["experiment"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    elif str(params["experiment"]) == "-1":
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()]
        if not subfolders:
            raise ValueError(f"Nessuna sottocartella trovata in {base_path}")
        subfolders = sorted(subfolders, key=lambda x: int(x))
        work_dir = os.path.join(base_path, subfolders[-1])
    else:
        work_dir = os.path.join(base_path, str(params["experiment"]))

    # Imposta semi
    seed = params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset e dataloader
    test_places = load_dataset(params["test_dataset"])
    dataset = TestDataset(test_places)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True, num_workers=8, persistent_workers=False)

    # Inizializza modello una sola volta
    model = MLPCosine(
        model_name=params["model"]["name"],
        trainable_from_layer=params["model"]["trainable_from_layer"],
        state_dict=params["model"]["state_dict"],
        device=device
    )
    model.to(device, non_blocking=True)
    model.eval()

    # Trova tutti i checkpoint nella cartella
    model_paths = glob(os.path.join(work_dir, "*.pt"))
    model_names = [os.path.split(path)[-1].split(".")[0] for path in model_paths]

    # Loop su ciascun checkpoint
    for j, path in enumerate(model_paths):
        model = MLPCosine.load_model_safely(model, path, device=device)
        model.to(device, non_blocking=True)
        model.eval()

        features0 = []
        features1 = []

        for mb in tqdm(dataloader):
            imgs0, imgs1 = mb  # ogni batch contiene tuple (img0, img1)
            f0 = model(imgs0.to(device))
            f1 = model(imgs1.to(device))
            features0.append(f0.detach().cpu())
            features1.append(f1.detach().cpu())

        features0 = torch.cat(features0)
        features1 = torch.cat(features1)

        cm0 = np.zeros((len(test_places), len(test_places)))
        cm1 = np.zeros((len(test_places), len(test_places)))

        compute_cm(features0, features1, cm0, j, "feat0")
        compute_cm(features1, features0, cm1, j, "feat1")
