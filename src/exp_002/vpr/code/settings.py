import os
import shutil
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_params, get_n_folders
from models import MLPCosine

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------ Work dir per train ------------------
def get_train_work_dir(params, experiments_dir="./experiments", config_file=None):
    # Usa "dataset" come base path
    base_path = os.path.join(experiments_dir, params.get("dataset", "run"))
    os.makedirs(base_path, exist_ok=True)

    # Se l'utente ha specificato work_dir nel YAML
    if params.get("work_dir") is not None:
        work_dir = os.path.join(base_path, str(params["work_dir"]))
    else:
        # Altrimenti crea nuova cartella numerata
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))

    os.makedirs(work_dir, exist_ok=True)

    # Copia il config file nella cartella
    if config_file:
        shutil.copy(config_file, work_dir)

    return work_dir


# ------------------ Work dir per test ------------------
def get_test_work_dir(params, experiments_dir="./experiments"):
    base_path = os.path.join(experiments_dir, params.get("save_dir", "run"))
    os.makedirs(base_path, exist_ok=True)

    experiment = params.get("experiment")

    if experiment is None:
        # Come default, crea nuova cartella numerata (non comune in test, ma ok come fallback)
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    elif str(experiment) == "-1":
        # Prendi l'ultima cartella numerata
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()]
        if not subfolders:
            raise ValueError(f"Nessuna sottocartella trovata in {base_path}")
        subfolders = sorted(subfolders, key=lambda x: int(x))
        work_dir = os.path.join(base_path, subfolders[-1])
    else:
        # Cartella specifica
        work_dir = os.path.join(base_path, str(experiment))

    if not os.path.exists(work_dir):
        raise ValueError(f"La cartella di test {work_dir} non esiste!")

    return work_dir

# ------------------ Inizializzazione modello ------------------
def init_model(params, device, load_state_dict=True):
    model_params = params["model"].copy()
    state_dict = model_params.pop("state_dict", None) if load_state_dict else None
    model = MLPCosine(device=device, **model_params)
    if state_dict:
        model.load_state_dict(torch.load(state_dict, map_location=device))
    model.to(device)
    return model

# ------------------ Ottimizzatore e scheduler ------------------
def init_optimizer_scheduler(model, params):
    lr = params["train"]["lr"]
    optimizer = _select_optimizer(model, params)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params["train"].get("lr_factor", 0.5),
        patience=params["train"].get("lr_patience", 5),
        min_lr=params["train"].get("lr_min", 1e-6),
        verbose=True
    ) if params["train"].get("reduce_lr_on_plateau", False) else None
    return optimizer, scheduler

def _select_optimizer(model, params):
    if params["train"]["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=params["train"]["lr"], weight_decay=params["train"].get("weight_decay", 0.0))
    elif params["train"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=params["train"]["lr"], weight_decay=params["train"].get("weight_decay", 0.0))


