##train.py

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
from torchmetrics.functional import accuracy

from loop import loop

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_n_folders, load_params, get_gta_places
from dataset import get_triplets, TriCombinationDataset
from models import MLPCosine

def build_dataset(triplets, params, shuffle=False):
    dataset = TriCombinationDataset(triplets, 
                                    use_center_crop=params["train"]["use_center_crop"],
                                    use_random_crop=params["train"]["use_random_crop"],
                                    normalize=params["train"]["normalize"])
    dataloader = DataLoader(dataset, params["train"]["batch_size"], shuffle=shuffle, drop_last=False, pin_memory=True, num_workers=8, persistent_workers=False)
    
    return dataloader

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    experiments_dir = "./experiments"
    
    config_file = "./train.yaml"
    params = load_params(config_file)
    
    base_path = os.path.join(experiments_dir, params["dataset"])
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
        os.makedirs(work_dir)
    else:
        work_dir = os.path.join(base_path, params["work_dir"])
        
    shutil.copy(config_file, work_dir)
    
    seed = params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)    
    
    n_epochs = params["train"]["n_epochs"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["lr"]   
    
    use_dataset = params["dataset"]
    
    valid_size = 25
        
    
    if use_dataset == "gta" or use_dataset == "all":
        gta_root = "/home/lcantagallo/VPR-GTAV2Real/src/exp_001_baseline/vpr/dataset/GTAV" #TODO change to the correct path
        gta_places = glob(os.path.join(gta_root, "*"))
        gta_places = [glob(os.path.join(p, "*.jpg")) for p in gta_places]
        gta_day_places_pre = get_gta_places(gta_places, "d_s")
        gta_night_places_pre = get_gta_places(gta_places, "n_c")
        
        gta_day_places = []
        gta_night_places = []
        
        for i, e in enumerate(gta_night_places_pre):
            if len(e) > 0:
                gta_day_places.append(gta_day_places_pre[i])
                gta_night_places.append(e)
            if len(gta_day_places) == 125:
                break

        gta_paths = [[gta_day_places[i][0], gta_night_places[i][np.random.randint(0, len(gta_night_places[i]))]] for i in range(len(gta_day_places))]

        gta_dataset_indices = np.arange(0, len(gta_paths))#np.arange(0, len(places))
        
        # per val = 0

        if valid_size == 0:
            gta_train_indices, gta_valid_indices = gta_dataset_indices, []
        else:
            gta_train_indices, gta_valid_indices = train_test_split(
                gta_dataset_indices, test_size=valid_size, random_state=seed
            )

        gta_train_indices = np.array(gta_train_indices, dtype=int)
        gta_valid_indices = np.array(gta_valid_indices, dtype=int)

        #fine val = 0
        
        print(f"dataset split: {len(gta_train_indices)}, {len(gta_valid_indices)}")
        dataset_to_save = {"train_idx": gta_train_indices.tolist(), "valid_idx": gta_valid_indices.tolist()}
        json_data = json.dumps(dataset_to_save)
        
        # Save JSON data to a file
        with open(os.path.join(work_dir, "gta_dataset.json"), "w") as json_file:
            json_file.write(json_data)  
    
    if use_dataset == "alderley" or use_dataset == "all":
        alderley_root = "/home/lcantagallo/VPR-GTAV2Real/src/exp_001_baseline/vpr/dataset/Alderley/alderley_paired" #TODO change to the correct path
        alderley_places = glob(os.path.join(alderley_root, "*"))
        alderley_places = [[alderley_places[i], alderley_places[i+1]] for i in range(len(alderley_places)) if i%2 == 0]
        
        alderley_dataset_indices = np.arange(0, len(alderley_places))#np.arange(0, len(places))
        
        if valid_size == 0:
            alderley_train_indices, alderley_valid_indices = alderley_dataset_indices, []
        else:
            alderley_train_indices, alderley_valid_indices = train_test_split(
                alderley_dataset_indices, test_size=valid_size, random_state=seed
            )

        alderley_train_indices = np.array(alderley_train_indices, dtype=int)
        alderley_valid_indices = np.array(alderley_valid_indices, dtype=int)

        
        print(f"dataset split: {len(alderley_train_indices)}, {len(alderley_valid_indices)}")
        dataset_to_save = {"train_idx": alderley_train_indices.tolist(), "valid_idx": alderley_valid_indices.tolist()}
        json_data = json.dumps(dataset_to_save)
        
        # Save JSON data to a file
        with open(os.path.join(work_dir, "alderley_dataset.json"), "w") as json_file:
            json_file.write(json_data)
    
    
    train_samples_per_place = params["train_samples_per_place"]
    valid_samples_per_place = params["valid_samples_per_place"]
    
    if use_dataset == "gta" or use_dataset == "all":
        gta_train_paths, gta_valid_paths = np.asarray(gta_paths)[gta_train_indices], np.asarray(gta_paths)[gta_valid_indices]
        gta_train_triplets, gta_valid_triplets = get_triplets(gta_train_paths, train_samples_per_place), get_triplets(gta_valid_paths, valid_samples_per_place)

    if use_dataset == "alderley" or use_dataset == "all":
        alderley_train_paths, alderley_valid_paths = np.asarray(alderley_places)[alderley_train_indices], np.asarray(alderley_places)[alderley_valid_indices]
        alderley_train_triplets, alderley_valid_triplets = get_triplets(alderley_train_paths, train_samples_per_place), get_triplets(alderley_valid_paths, valid_samples_per_place)
    
    if use_dataset == "gta":
        train_triplets = gta_train_triplets
        valid_triplets = gta_valid_triplets
    elif use_dataset == "alderley":
        train_triplets = alderley_train_triplets
        valid_triplets = alderley_valid_triplets
    elif use_dataset == "all":
        train_triplets = [*gta_train_triplets, *alderley_train_triplets]
        valid_triplets = [*gta_valid_triplets, *alderley_valid_triplets]        
        
    train_dataloader = build_dataset(train_triplets, params, True)
    valid_dataloader = build_dataset(valid_triplets, params, False)    
    
   
    model = MLPCosine(device=device, **params["model"])

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    fraction = num_trainable_params / num_total_params

    print(f"Trainable parameters: {num_trainable_params:,}")
    print(f"Total parameters: {num_total_params:,}")
    print(f"Fraction trainable: {fraction:.3f}")
   
    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_function)
        
    writer = SummaryWriter(log_dir=work_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    min_loss = 1e10  
    for t in range(n_epochs):
        print(f"Epoch {t+1}")
        train_loss = loop(model,
                        train_dataloader,
                        loss_fn,
                        optimizer,
                        train=True,
                        device=device)
        
        writer.add_scalar("train/loss", train_loss, t)        
        valid_loss = loop(model,
                        valid_dataloader,
                        loss_fn,
                        optimizer=None,
                        train=False,
                        device=device)
        
        writer.add_scalar("valid/loss", valid_loss, t)
        with open(os.path.join(work_dir, "training_log.txt"), "a") as log_file:
            log_file.write(f"{t+1}\t{train_loss:.6f}\t{valid_loss:.6f}\n")

        
        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(work_dir, "best_loss.pt"))
        
        if use_dataset == "gta" or use_dataset == "all":
            gta_train_triplets, gta_valid_triplets = get_triplets(gta_train_paths, train_samples_per_place), get_triplets(gta_valid_paths, valid_samples_per_place)
        
        if use_dataset == "alderley" or use_dataset == "all":
            alderley_train_triplets, alderley_valid_triplets = get_triplets(alderley_train_paths, train_samples_per_place), get_triplets(alderley_valid_paths, valid_samples_per_place)
        
        if use_dataset == "gta":
            train_triplets = gta_train_triplets
            valid_triplets = gta_valid_triplets
        elif use_dataset == "alderley":
            train_triplets = alderley_train_triplets
            valid_triplets = alderley_valid_triplets
        elif use_dataset == "all":
            train_triplets = [*gta_train_triplets, *alderley_train_triplets]
            valid_triplets = [*gta_valid_triplets, *alderley_valid_triplets] 
        
        train_dataloader = build_dataset(train_triplets, params, True)
        valid_dataloader = build_dataset(valid_triplets, params, False)
    
    torch.save(model.state_dict(), os.path.join(work_dir, "end.pt"))