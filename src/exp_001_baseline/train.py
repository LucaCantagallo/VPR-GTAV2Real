import os
import torch
from torch import nn
import shutil
import json
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time

from loop import loop
from logger import ExperimentLogger 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_n_folders, load_params, get_gta_places
from dataset import get_triplets, TriCombinationDataset
from models import MLPCosine

def build_dataset(triplets, params, shuffle=False):
    dataset = TriCombinationDataset(
        triplets, 
        use_center_crop=params["train"]["use_center_crop"],
        use_random_crop=params["train"]["use_random_crop"],
        normalize=params["train"]["normalize"]
    )
    dataloader = DataLoader(
        dataset, params["train"]["batch_size"], shuffle=shuffle, 
        drop_last=False, pin_memory=True, num_workers=8, persistent_workers=False
    )
    return dataloader

if __name__ == "__main__":
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # ------------------- CONFIG & WORK DIR -------------------
    experiments_dir = "./experiments"
    config_file = "./train.yaml"
    params = load_params(config_file)
    
    base_path = os.path.join(experiments_dir, params["dataset"])
    os.makedirs(base_path, exist_ok=True)
    
    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
        os.makedirs(work_dir)
    else:
        work_dir = os.path.join(base_path, params["work_dir"])
    
    shutil.copy(config_file, work_dir)

    # ------------------- SEED -------------------
    seed = params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_epochs = params["train"]["n_epochs"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["lr"]
    use_dataset = params["dataset"]
    valid_size = 50

    # ------------------- DATASET SPLIT -------------------
    if use_dataset in ["gta", "all"]:
        gta_root = "/home/lcantagallo/VPR-GTAV2Real/dataset/GTAV"
        gta_places = glob(os.path.join(gta_root, "*"))
        gta_places = [glob(os.path.join(p, "*.jpg")) for p in gta_places]
        gta_day_places_pre = get_gta_places(gta_places, "d_s")
        gta_night_places_pre = get_gta_places(gta_places, "n_c")

        gta_day_places, gta_night_places = [], []
        for i, e in enumerate(gta_night_places_pre):
            if len(e) > 0:
                gta_day_places.append(gta_day_places_pre[i])
                gta_night_places.append(e)
            if len(gta_day_places) == 125:
                break

        gta_paths = [
            [gta_day_places[i][0], gta_night_places[i][np.random.randint(0, len(gta_night_places[i]))]] 
            for i in range(len(gta_day_places))
        ]

        gta_dataset_indices = np.arange(len(gta_paths))
        gta_train_indices, gta_valid_indices = train_test_split(gta_dataset_indices, test_size=valid_size, random_state=seed)
        print(f"dataset split: {len(gta_train_indices)}, {len(gta_valid_indices)}")
        json.dump({"train_idx": gta_train_indices.tolist(), "valid_idx": gta_valid_indices.tolist()}, 
                  open(os.path.join(work_dir, "gta_dataset.json"), "w"))

    if use_dataset in ["alderley", "all"]:
        alderley_root = "/home/lcantagallo/VPR-GTAV2Real/dataset/Alderley/alderley_paired"
        alderley_places = glob(os.path.join(alderley_root, "*"))
        alderley_places = [[alderley_places[i], alderley_places[i+1]] for i in range(len(alderley_places)) if i % 2 == 0]

        alderley_dataset_indices = np.arange(len(alderley_places))
        alderley_train_indices, alderley_valid_indices = train_test_split(alderley_dataset_indices, test_size=valid_size, random_state=seed)
        print(f"dataset split: {len(alderley_train_indices)}, {len(alderley_valid_indices)}")
        json.dump({"train_idx": alderley_train_indices.tolist(), "valid_idx": alderley_valid_indices.tolist()}, 
                  open(os.path.join(work_dir, "alderley_dataset.json"), "w"))

    train_samples_per_place = params["train_samples_per_place"]
    valid_samples_per_place = params["valid_samples_per_place"]

    if use_dataset in ["gta", "all"]:
        gta_train_paths, gta_valid_paths = np.asarray(gta_paths)[gta_train_indices], np.asarray(gta_paths)[gta_valid_indices]
        gta_train_triplets = get_triplets(gta_train_paths, train_samples_per_place)
        gta_valid_triplets = get_triplets(gta_valid_paths, valid_samples_per_place)

    if use_dataset in ["alderley", "all"]:
        alderley_train_paths, alderley_valid_paths = np.asarray(alderley_places)[alderley_train_indices], np.asarray(alderley_places)[alderley_valid_indices]
        alderley_train_triplets = get_triplets(alderley_train_paths, train_samples_per_place)
        alderley_valid_triplets = get_triplets(alderley_valid_paths, valid_samples_per_place)

    if use_dataset == "gta":
        train_triplets, valid_triplets = gta_train_triplets, gta_valid_triplets
    elif use_dataset == "alderley":
        train_triplets, valid_triplets = alderley_train_triplets, alderley_valid_triplets
    elif use_dataset == "all":
        train_triplets = [*gta_train_triplets, *alderley_train_triplets]
        valid_triplets = [*gta_valid_triplets, *alderley_valid_triplets]

    train_dataloader = build_dataset(train_triplets, params, shuffle=True)
    valid_dataloader = build_dataset(valid_triplets, params, shuffle=False)

    # ------------------- MODEL & LOSS -------------------
    model = MLPCosine(device=device, **params["model"])
    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=work_dir)

    # ------------------- LOGGER -------------------
    logger = ExperimentLogger(
        work_dir=work_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        dataset_name=use_dataset,
        device=device
    )

    min_loss = 1e10

    # ------------------- TRAINING LOOP -------------------
    for t in range(n_epochs):
        print(f"Epoch {t+1}")

        # Train
        train_loss = loop(model, train_dataloader, loss_fn, optimizer, train=True, device=device)
        writer.add_scalar("train/loss", train_loss, t)

        # Validation
        valid_loss = loop(model, valid_dataloader, loss_fn, optimizer=None, train=False, device=device)
        writer.add_scalar("valid/loss", valid_loss, t)

        # Update logger
        is_best = logger.log_epoch(epoch=t+1, train_loss=train_loss, valid_loss=valid_loss)
        if is_best:
            torch.save(model.state_dict(), os.path.join(work_dir, "best_loss.pt"))

        # Rigenera triplets e dataloader se serve
        if use_dataset in ["gta", "all"]:
            gta_train_triplets = get_triplets(gta_train_paths, train_samples_per_place)
            gta_valid_triplets = get_triplets(gta_valid_paths, valid_samples_per_place)
        if use_dataset in ["alderley", "all"]:
            alderley_train_triplets = get_triplets(alderley_train_paths, train_samples_per_place)
            alderley_valid_triplets = get_triplets(alderley_valid_paths, valid_samples_per_place)

        if use_dataset == "gta":
            train_triplets, valid_triplets = gta_train_triplets, gta_valid_triplets
        elif use_dataset == "alderley":
            train_triplets, valid_triplets = alderley_train_triplets, alderley_valid_triplets
        elif use_dataset == "all":
            train_triplets = [*gta_train_triplets, *alderley_train_triplets]
            valid_triplets = [*gta_valid_triplets, *alderley_valid_triplets]

        train_dataloader = build_dataset(train_triplets, params, shuffle=True)
        valid_dataloader = build_dataset(valid_triplets, params, shuffle=False)

    # ------------------- END TRAINING -------------------
    total_time = time.time() - start_time
    print(f"Training completato in {total_time/60:.2f} minuti")
    logger.set_total_time(total_time)
    logger.save_summary()

    # Save final model
    torch.save(model.state_dict(), os.path.join(work_dir, "end.pt"))
