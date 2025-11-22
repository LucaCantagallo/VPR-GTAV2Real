import os, shutil, json, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loop import loop
from models import MLPCosine
from utils import get_n_folders, load_params
from triplet_loader import get_dataloaders, refresh_dataloaders, get_triplet_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    experiments_dir = "./experiments"
    config_file = "./pipeline.yaml"
    params = load_params(config_file)
    
    base_path = os.path.join(experiments_dir, params.get("dataset", "run"))
    os.makedirs(base_path, exist_ok=True)
    work_dir = os.path.join(base_path, str(get_n_folders(base_path))) if params.get("work_dir") is None else os.path.join(base_path, params["work_dir"])
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(config_file, work_dir)
    
    seed = params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_epochs = params["train"]["n_epochs"]
    lr = params["train"]["lr"]
    
    train_loader, valid_loader, train_places, valid_places = get_dataloaders(
        params["dataload"], params["train_dataset"], params["val_dataset"],
        params["train_samples_per_place"], params["valid_samples_per_place"], params, seed
    )
    
    model = MLPCosine(device=device, **params["model"])
    
    loss_fn = get_triplet_loss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=params["train"].get("weight_decay", 0.0))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["train"].get("lr_factor", 0.5),
                                  patience=params["train"].get("lr_patience", 5), min_lr=params["train"].get("lr_min", 1.e-6),
                                  verbose=True) if params["train"].get("reduce_lr_on_plateau", False) else None
    
    writer = SummaryWriter(log_dir=work_dir)
    min_loss, best_epoch, no_improve_counter = float("inf"), 0, 0
    epoch_history = []
    
    for epoch in range(n_epochs):
        train_loss = loop(model, train_loader, loss_fn, optimizer, train=True, device=device)
        valid_loss = loop(model, valid_loader, loss_fn, optimizer=None, train=False, device=device)
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("valid/loss", valid_loss, epoch)
        
        if scheduler:
            scheduler.step(valid_loss)
        
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_epoch = epoch + 1
            no_improve_counter = 0
            torch.save(model.state_dict(), os.path.join(work_dir, "best_loss.pt"))
        else:
            no_improve_counter += 1
        
        if params["train"].get("early_stopping", False) and no_improve_counter >= params["train"].get("patience", 0):
            break
        
        # --- rigenera triplets ogni epoca
        train_loader, valid_loader = refresh_dataloaders(train_places, valid_places,
                                                         params["train_samples_per_place"],
                                                         params["valid_samples_per_place"], params)
    
    with open(os.path.join(work_dir, "training_summary.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_loss": float(min_loss), "total_epochs": epoch+1}, f, indent=4)
