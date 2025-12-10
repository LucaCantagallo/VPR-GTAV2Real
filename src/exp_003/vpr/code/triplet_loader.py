import numpy as np
from sklearn.model_selection import train_test_split
from itertools import permutations, combinations
from dataset import BaseDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from places_extractor import extract_places
from filter_loader_daynight import filter_paired_daynight
from filter_loader_vpr import filter_paired_vpr

# ---------------------------
# DATASET AGGIORNATO
# ---------------------------
class TriCombinationDataset(BaseDataset):
    def __init__(self, triplets, params):
        super().__init__(paths=triplets, params=params)

    def __getitem__(self, index):
        paths = self.paths[index] # paths è una tripla [anchor, pos, neg]
        
        anchor, crop = self.__load__(paths[0])
        positive, _ = self.__load__(paths[1], crop) 
        negative, _ = self.__load__(paths[2])
        
        return anchor, positive, negative

# ---------------------------
# LOADERS PUBBLICI
# ---------------------------

def get_triplet_dataloaders(dataload_mode, train_dataset, val_dataset, train_samples_per_place, valid_samples_per_place, params, seed):
    train_places_paired, valid_places_paired = _load_and_split(dataload_mode, train_dataset, val_dataset, seed)
    train_triplets = _generate_triplets(train_places_paired, train_samples_per_place)
    valid_triplets = _generate_triplets(valid_places_paired, valid_samples_per_place)
    
    preprocessing_params = params["train"].get("preprocessing", {})

    train_loader = DataLoader(
        TriCombinationDataset(
            triplets=train_triplets, 
            params=preprocessing_params
        ),
        batch_size=params["train"]["batch_size"], shuffle=True, num_workers=8
    )
    
    valid_loader = DataLoader(
        TriCombinationDataset(
            triplets=valid_triplets,
            params=preprocessing_params
        ),
        batch_size=params["train"]["batch_size"], shuffle=False, num_workers=8
    )
    
    return train_loader, valid_loader, train_places_paired, valid_places_paired

def refresh_triplet_dataloaders(train_places, valid_places, train_samples_per_place, valid_samples_per_place, params):
    train_triplets = _generate_triplets(train_places, train_samples_per_place)
    valid_triplets = _generate_triplets(valid_places, valid_samples_per_place)
    
    preprocessing_params = params["train"].get("preprocessing", {})

    train_loader = DataLoader(
        TriCombinationDataset(
            triplets=train_triplets, 
            params=preprocessing_params
        ),
        batch_size=params["train"]["batch_size"], shuffle=True, num_workers=8
    )
    
    valid_loader = DataLoader(
        TriCombinationDataset(
            triplets=valid_triplets,
            params=preprocessing_params
        ),
        batch_size=params["train"]["batch_size"], shuffle=False, num_workers=8
    )
    
    return train_loader, valid_loader


def run_triplet_epoch(model,
         dataloader,
         loss_fn,
         optimizer=None,
         train=True,
         device="cpu"
         ):
    if train:
        model.train()
    else:
        model.eval()
    loss_val = 0.0
    
    with torch.set_grad_enabled(train):
        for mb_idx, mb in enumerate(tqdm(dataloader)):            
            
            anchor, positive, negative = mb
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor), model(positive), model(negative)
            loss = loss_fn(anchor_output, positive_output, negative_output) 

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            loss_val += loss.item()            
            
    loss_val /= (mb_idx + 1)
    return loss_val

def get_triplet_loss():
    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    return nn.TripletMarginWithDistanceLoss(distance_function=distance_function)

def test_paired_loader(dataload_mode, test_dataset):
    test_places = extract_places(test_dataset)
    
    if dataload_mode == "daynight":
        test_places = filter_paired_daynight(test_dataset, test_places)
    else:
        test_places = filter_paired_vpr(test_dataset, test_places)
    return test_places



# --- private
def _load_and_split(dataload_mode, train_dataset, val_dataset, seed):
    train_places = extract_places(train_dataset)
    valid_places = extract_places(val_dataset)

    if dataload_mode == "daynight":
        train_places = filter_paired_daynight(train_dataset, train_places)
        valid_places = filter_paired_daynight(val_dataset, valid_places)
    
 
    if train_dataset == val_dataset:
        indices = np.arange(len(train_places))
        train_idx, valid_idx = train_test_split(indices, test_size=0.25, random_state=seed, shuffle=True)
        train_places = [train_places[i] for i in train_idx]
        valid_places = [valid_places[i] for i in valid_idx]
    
    return train_places, valid_places


def _generate_triplets(places, samples_per_place=3, use_combination=False):
    triplets = []

    for place_idx, place_images in enumerate(places):
        num_images = len(place_images)
        if num_images < 2:
            continue

        if use_combination:
            pairs_idx = list(combinations(range(num_images), 2))
        else:
            pairs_idx = list(permutations(range(num_images), 2))

        if samples_per_place > 0 and len(pairs_idx) > samples_per_place:
            selected_idx = np.random.choice(len(pairs_idx), samples_per_place, replace=False)
            pairs_idx = [pairs_idx[i] for i in selected_idx]

        for anchor_idx, positive_idx in pairs_idx:
            anchor = place_images[anchor_idx]
            positive = place_images[positive_idx]

            while True:
                neg_place_idx = np.random.randint(0, len(places))
                if neg_place_idx != place_idx and len(places[neg_place_idx]) > 0:
                    negative = places[neg_place_idx][np.random.randint(len(places[neg_place_idx]))]
                    break

            triplets.append([str(anchor), str(positive), str(negative)])

    return triplets