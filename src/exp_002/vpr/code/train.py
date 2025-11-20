## train.py

import os
import torch
from torch import nn
import shutil
import json
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loop import loop
from dataset import get_triplets, TriCombinationDataset
from models import MLPCosine


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_n_folders, load_params




def build_dataset(triplets, params, shuffle=False):
    dataset = TriCombinationDataset(triplets, 
                                    use_center_crop=params["train"]["use_center_crop"],
                                    use_random_crop=params["train"]["use_random_crop"],
                                    normalize=params["train"]["normalize"])
    dataloader = DataLoader(dataset, params["train"]["batch_size"], shuffle=shuffle,
                            drop_last=False, pin_memory=True, num_workers=8, persistent_workers=False)
    return dataloader

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


    base_path = os.path.join(experiments_dir, params.get("dataset", "run"))
    os.makedirs(base_path, exist_ok=True)
        
    work_dir = os.path.join(base_path, str(get_n_folders(base_path))) if params.get("work_dir", None) is None else os.path.join(base_path, params["work_dir"])
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(config_file, work_dir)

    seed = params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_epochs = params["train"]["n_epochs"]
    lr = params["train"]["lr"]
    use_early_stopping = params["train"].get("early_stopping", False)
    patience = params["train"].get("patience", 0)
    min_delta = float(params["train"].get("min_delta", 0.0))
    weight_decay = params["train"].get("weight_decay", 0.0)
    use_reduce_on_plateau = params["train"].get("reduce_lr_on_plateau", False)
    opt_name = params["train"].get("optimizer", "adamw").lower()
    
    train_dataset = params["train_dataset"]
    val_dataset = params["val_dataset"]

    # --- carica dataset completo tramite data_loader.py (lista di places)
    train_places = load_dataset(train_dataset)
    valid_places = load_dataset(val_dataset)

    # --- se sono lo stesso dataset, fai split 75/25 a livello di PLACE
    if train_dataset == val_dataset:
        # split in modo deterministico basato sul seed
        indices = np.arange(len(train_places))
        train_idx, valid_idx = train_test_split(indices, test_size=0.25, random_state=seed, shuffle=True)
        train_places = [train_places[i] for i in train_idx]
        valid_places = [valid_places[i] for i in valid_idx]

    # --- genera triplette (passando LISTE di places — non converto in np.array per evitare dtype=object issues)
    train_triplets = get_triplets(train_places, params["train_samples_per_place"])
    valid_triplets = get_triplets(valid_places, params["valid_samples_per_place"])

    # --- costruisci DataLoader
    train_dataloader = build_dataset(train_triplets, params, shuffle=True)
    valid_dataloader = build_dataset(valid_triplets, params, shuffle=False)

    # --- model + training setup (identico alla v1)
    model = MLPCosine(device=device, **params["model"])

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable_params:,}, Total: {num_total_params:,}, Fraction: {num_trainable_params/num_total_params:.3f}")

    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_function)

    writer = SummaryWriter(log_dir=work_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if opt_name == "adamw" else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["train"].get("lr_factor", 0.5),
                                  patience=params["train"].get("lr_patience", 5), min_lr=params["train"].get("lr_min", 1.e-6), verbose=True) if use_reduce_on_plateau else None

    no_improve_counter = 0
    best_epoch = 0
    min_loss = float("inf")
    epoch_history = []

    # salva anche gli indices usati (utile per riproducibilità) — facoltativo ma utile come in v1
    try:
        with open(os.path.join(work_dir, "dataset_split.json"), "w") as f:
            json.dump({
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "train_count": len(train_places),
                "valid_count": len(valid_places)
            }, f, indent=2)
    except Exception:
        pass

    for t in range(n_epochs):
        print(f"Epoch {t+1}")
        train_loss = loop(model, train_dataloader, loss_fn, optimizer, train=True, device=device)
        writer.add_scalar("train/loss", train_loss, t)

        valid_loss = 0.0 if len(valid_dataloader.dataset) == 0 else loop(model, valid_dataloader, loss_fn, optimizer=None, train=False, device=device)
        writer.add_scalar("valid/loss", valid_loss, t)

        with open(os.path.join(work_dir, "training_log.txt"), "a") as log_file:
            log_file.write(f"{t+1}\t{train_loss:.6f}\t{valid_loss:.6f}\n")
            if use_reduce_on_plateau:
                log_file.write(f"lr_epoch_{t+1}: {optimizer.param_groups[0]['lr']}\n")

        if use_reduce_on_plateau:
            scheduler.step(valid_loss)

        epoch_history.append({"epoch": t+1, "train_loss": float(train_loss), "valid_loss": float(valid_loss), "lr": optimizer.param_groups[0]['lr']})

        if valid_loss < min_loss - min_delta:
            min_loss = valid_loss
            best_epoch = t + 1
            no_improve_counter = 0
            torch.save(model.state_dict(), os.path.join(work_dir, "best_loss.pt"))
        else:
            no_improve_counter += 1

        if use_early_stopping and no_improve_counter >= patience:
            print(f"Early stopping at epoch {t+1}")
            break

        # --- ricrea triplets ogni epoca (comportamento v1)
        train_triplets = get_triplets(train_places, params["train_samples_per_place"]) if len(train_places) > 0 else []
        valid_triplets = get_triplets(valid_places, params["valid_samples_per_place"]) if len(valid_places) > 0 else []
        train_dataloader = build_dataset(train_triplets, params, shuffle=True)
        valid_dataloader = build_dataset(valid_triplets, params, shuffle=False)

    summary = {"best_epoch": best_epoch, "best_loss": float(min_loss), "total_epochs": t+1,
               "early_stopping_used": use_early_stopping, "patience": patience, "epochs": epoch_history}

    with open(os.path.join(work_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
