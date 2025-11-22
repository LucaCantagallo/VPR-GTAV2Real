from data_loader import dataload_paired
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import permutations, combinations
from dataset import BaseDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- pubblica: restituisce DataLoader pronti per training/validazione
def get_dataloaders(dataload_mode, train_dataset, val_dataset, train_samples_per_place, valid_samples_per_place, params, seed):
    train_places_paired, valid_places_paired = _load_and_split(dataload_mode, train_dataset, val_dataset, seed)
    train_triplets = _generate_triplets(train_places_paired, train_samples_per_place)
    valid_triplets = _generate_triplets(valid_places_paired, valid_samples_per_place)
    
    train_loader = DataLoader(TriCombinationDataset(train_triplets, 
                                                    use_center_crop=params["train"]["use_center_crop"],
                                                    use_random_crop=params["train"]["use_random_crop"],
                                                    normalize=params["train"]["normalize"]),
                              batch_size=params["train"]["batch_size"], shuffle=True, num_workers=8)
    
    valid_loader = DataLoader(TriCombinationDataset(valid_triplets,
                                                    use_center_crop=params["train"]["use_center_crop"],
                                                    use_random_crop=params["train"]["use_random_crop"],
                                                    normalize=params["train"]["normalize"]),
                              batch_size=params["train"]["batch_size"], shuffle=False, num_workers=8)
    
    return train_loader, valid_loader, train_places_paired, valid_places_paired


# --- pubblica: rigenera DataLoader a ogni epoca (stesso formato)
def refresh_dataloaders(train_places, valid_places, train_samples_per_place, valid_samples_per_place, params):
    train_triplets = _generate_triplets(train_places, train_samples_per_place)
    valid_triplets = _generate_triplets(valid_places, valid_samples_per_place)
    
    train_loader = DataLoader(TriCombinationDataset(train_triplets, 
                                                    use_center_crop=params["train"]["use_center_crop"],
                                                    use_random_crop=params["train"]["use_random_crop"],
                                                    normalize=params["train"]["normalize"]),
                              batch_size=params["train"]["batch_size"], shuffle=True, num_workers=8)
    
    valid_loader = DataLoader(TriCombinationDataset(valid_triplets,
                                                    use_center_crop=params["train"]["use_center_crop"],
                                                    use_random_crop=params["train"]["use_random_crop"],
                                                    normalize=params["train"]["normalize"]),
                              batch_size=params["train"]["batch_size"], shuffle=False, num_workers=8)
    
    return train_loader, valid_loader

def get_triplet_loss():
    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    return nn.TripletMarginWithDistanceLoss(distance_function=distance_function)


# --- private
def _load_and_split(dataload_mode, train_dataset, val_dataset, seed):
    train_places_paired = dataload_paired(dataload_mode, train_dataset)
    valid_places_paired = dataload_paired(dataload_mode, val_dataset)
    
    if train_dataset == val_dataset:
        indices = np.arange(len(train_places_paired))
        train_idx, valid_idx = train_test_split(indices, test_size=0.25, random_state=seed, shuffle=True)
        train_places_paired = [train_places_paired[i] for i in train_idx]
        valid_places_paired = [valid_places_paired[i] for i in valid_idx]
    
    return train_places_paired, valid_places_paired


def _generate_triplets(places, samples_per_place=-1, use_combination=False):
    triplets = []
    indices = []
    for i, path in enumerate(places):
        idx = np.arange(len(path))
        if use_combination:
            pair = list(map(list, combinations(idx, 2)))
        else:
            pair = list(map(list, permutations(idx, 2)))
        indices.append(pair)

    for i, idx in enumerate(indices):
        if samples_per_place > 0 and len(idx) > samples_per_place:
            selected = np.random.choice(len(idx), samples_per_place, False)
            current_triplets = np.array(places[i])[idx][selected].tolist()
        else:
            current_triplets = np.array(places[i])[idx].tolist()

        for j, elem in enumerate(current_triplets):
            while True:
                negative_idx = np.random.randint(0, len(places))
                if negative_idx != i:
                    if len(idx) == 2:
                        negative_sample_idx = idx[j][1]
                    else:
                        negative_sample_idx = np.random.randint(len(places[negative_idx]))
                    elem.append(str(places[negative_idx][negative_sample_idx]))
                    break
        triplets.extend(current_triplets)
    return triplets

class TriCombinationDataset(BaseDataset):
    def __getitem__(self, index):
        paths = self.paths[index]
        anchor, crop = self.__load__(paths[0])
        positive, _ = self.__load__(paths[1], crop)
        negative, _ = self.__load__(paths[2])
        return anchor, positive, negative


