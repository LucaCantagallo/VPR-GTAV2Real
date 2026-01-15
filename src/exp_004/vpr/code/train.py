### train.py
import os, shutil, json, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import VPRModel
from utils import get_n_folders, load_params
from learning_strategy import get_dataloaders, run_epoch, run_epoch_valid, refresh_dataloaders, get_loss_fn



from settings import get_device, get_train_work_dir, init_optimizer_scheduler, set_seed, init_model

if __name__ == "__main__":
    device = get_device()
    
    experiments_dir = "./experiments"
    config_file = "./pipeline.yaml"
    params = load_params(config_file)
    
    # Inizializza cartella di lavoro (train)
    work_dir = get_train_work_dir(params, experiments_dir=experiments_dir, config_file=config_file)
    
    set_seed(params["seed"])
    
    n_epochs = params["train"]["n_epochs"]
    lr = params["train"]["lr"]

    train_loader, valid_loader, train_places, valid_places= get_dataloaders(params)
    loss_fn = get_loss_fn(params)

    
    model = init_model(params, device)
    
    
        
    optimizer, scheduler = init_optimizer_scheduler(model, params)

    writer = SummaryWriter(log_dir=work_dir)

    min_loss, best_epoch, no_improve_counter = float("inf"), 0, 0
    epoch_history = []
    use_early_stopping = params["train"].get("early_stopping", False)
    patience = params["train"].get("patience", 0)
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")

        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, params, device)
        valid_loss = run_epoch_valid(model, valid_loader, loss_fn, params, device)
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("valid/loss", valid_loss, epoch)
        
        if scheduler:
            scheduler.step(valid_loss)

        # salva la cronologia epoca
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "lr": float(optimizer.param_groups[0]['lr'])
        })
        
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_epoch = epoch + 1
            no_improve_counter = 0
            torch.save(model.state_dict(), os.path.join(work_dir, "best_loss.pt"))
        else:
            no_improve_counter += 1
        
        if use_early_stopping and no_improve_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # --- rigenera triplets ogni epoca
        train_loader, valid_loader = refresh_dataloaders(train_places, valid_places, params)
        
    # salva il JSON dettagliato
    summary = {
        "best_epoch": best_epoch,
        "best_loss": float(min_loss),
        "total_epochs": epoch + 1,
        "early_stopping_used": use_early_stopping,
        "patience": patience,
        "epochs": epoch_history
    }
    
    with open(os.path.join(work_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    