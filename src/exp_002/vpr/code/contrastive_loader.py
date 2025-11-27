# contrastive_loader.py

import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import BaseDataset
from places_extractor import extract_places
from filter_loader_daynight import filter_paired_daynight
from filter_loader_vpr import filter_paired_vpr
from sklearn.model_selection import train_test_split
from itertools import combinations
from tqdm import tqdm


# ---------------------------
# DATASET
# ---------------------------
class SupConDataset(BaseDataset):
    def __init__(self, batches, target_width=224, target_height=224,
                 use_center_crop=False, use_random_crop=False, normalize=False,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.batches = batches
        super().__init__(paths=[], target_width=target_width, target_height=target_height,
                         use_center_crop=use_center_crop, use_random_crop=use_random_crop,
                         normalize=normalize, mean=mean, std=std)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]   # lista di (path, label)

        imgs = []
        labels = []

        for path, label in batch:
            img, _ = self.__load__(path, crop=None)
            imgs.append(img)
            labels.append(label)

        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)



# ---------------------------
# GENERAZIONE BATCH PER PLACE
# ---------------------------
def _load_and_split(dataload_mode, train_dataset, val_dataset, seed):
    train_places = extract_places(train_dataset)
    valid_places = extract_places(val_dataset)

    if dataload_mode == "daynight":
        train_places = filter_paired_daynight(train_dataset, train_places)
        valid_places = filter_paired_daynight(val_dataset, valid_places)

    if train_dataset == val_dataset:
        idx = np.arange(len(train_places))
        tr, va = train_test_split(idx, test_size=0.25, random_state=seed, shuffle=True)
        train_places = [train_places[i] for i in tr]
        valid_places = [valid_places[i] for i in va]

    return train_places, valid_places


def _generate_supcon_batches(places, samples_per_place, places_per_batch=1):
    all_batches = []
    num_places = len(places)

    for _ in range(num_places // places_per_batch):
        # campiona nuovi posti per questo batch
        batch_places_idx = np.random.choice(num_places, places_per_batch, replace=False)

        batch = []
        for place_idx in batch_places_idx:
            images = places[place_idx].copy()
            np.random.shuffle(images)

            if len(images) > samples_per_place:
                sel = np.random.choice(len(images), samples_per_place, replace=False)
                sel_imgs = [images[j] for j in sel]
            else:
                sel_imgs = images

            batch += [(p, place_idx) for p in sel_imgs]

        all_batches.append(batch)

    return all_batches





# ---------------------------
# PUBLIC
# ---------------------------
def get_contrastive_dataloaders(dataload_mode, train_dataset, val_dataset,
                                train_samples_per_place, valid_samples_per_place,
                                params, seed):

    train_places, valid_places = _load_and_split(dataload_mode, train_dataset, val_dataset, seed)

    places_per_batch = params.get("contrastive_places_per_batch", 1)
    train_batches = _generate_supcon_batches(train_places, train_samples_per_place, places_per_batch)
    valid_batches = _generate_supcon_batches(valid_places, valid_samples_per_place, places_per_batch)


    train_loader = DataLoader(
        SupConDataset(train_batches,
                      use_center_crop=params["train"]["use_center_crop"],
                      use_random_crop=params["train"]["use_random_crop"],
                      normalize=params["train"]["normalize"]),
        batch_size=1, shuffle=True, num_workers=8, collate_fn=lambda x: x[0]
    )

    valid_loader = DataLoader(
        SupConDataset(valid_batches,
                      use_center_crop=params["train"]["use_center_crop"],
                      use_random_crop=params["train"]["use_random_crop"],
                      normalize=params["train"]["normalize"]),
        batch_size=1, shuffle=False, num_workers=8, collate_fn=lambda x: x[0]
    )

    return train_loader, valid_loader, train_places, valid_places


def refresh_contrastive_dataloaders(train_places, valid_places,
                                    train_samples_per_place, valid_samples_per_place, params):

    train_batches = _generate_supcon_batches(train_places, train_samples_per_place, params["contrastive_places_per_batch"])
    valid_batches = _generate_supcon_batches(valid_places, valid_samples_per_place, params["contrastive_places_per_batch"])

    train_loader = DataLoader(
        SupConDataset(train_batches,
                      use_center_crop=params["train"]["use_center_crop"],
                      use_random_crop=params["train"]["use_random_crop"],
                      normalize=params["train"]["normalize"]),
        batch_size=1, shuffle=True, num_workers=8, collate_fn=lambda x: x[0]
    )

    valid_loader = DataLoader(
        SupConDataset(valid_batches,
                      use_center_crop=params["train"]["use_center_crop"],
                      use_random_crop=params["train"]["use_random_crop"],
                      normalize=params["train"]["normalize"]),
        batch_size=1, shuffle=False, num_workers=8, collate_fn=lambda x: x[0]
    )

    return train_loader, valid_loader


# ---------------------------
# LOSS NT-XENT (SUPCON)
# ---------------------------
def get_supcon_loss(temperature=0.07):
    def loss_fn(features, labels):
        B = features.size(0)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask = mask - torch.eye(B, device=features.device)

        sim_exp = torch.exp(sim)
        denom = sim_exp.sum(dim=1, keepdim=True)
        log_prob = sim - torch.log(denom + 1e-8)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob.mean()

        return loss

    return loss_fn


# ---------------------------
# TRAINER
# ---------------------------
def run_supcon_epoch(model, dataloader, loss_fn, optimizer=None, train=True, device="cpu"):
    if train:
        model.train()
    else:
        model.eval()

    losses = []

    dataloader_iter = tqdm(dataloader, total=len(dataloader), ncols=80)
    for batch in dataloader_iter:
        imgs, labels = batch

        # Diagnostics: ensure expected shapes
        if imgs.dim() != 4:
            raise ValueError(f"Expected imgs with 4 dims [N,C,H,W], got {imgs.shape}")
        if labels.dim() != 1:
            raise ValueError(f"Expected labels 1D tensor [N], got {labels.shape}")


        imgs = imgs.to(device)
        labels = labels.to(device)

        feats = model(imgs)

        loss = loss_fn(feats, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        dataloader_iter.set_postfix(loss=loss.item())

    return sum(losses) / len(losses)
