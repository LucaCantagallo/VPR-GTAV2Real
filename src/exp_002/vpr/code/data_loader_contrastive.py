import numpy as np
from sklearn.model_selection import train_test_split
from dataset import BaseDataset
from torch.utils.data import DataLoader
import torch

from places_extractor import extract_places

# --- pubblica: restituisce DataLoader pronti per training/validazione
def get_dataloaders(dataload_mode, train_dataset, val_dataset, params, seed):
    if dataload_mode == "daynight":
        raise ValueError("Contrastive learning non supportato per day/night datasets")

    train_places, valid_places = _load_and_split(train_dataset, val_dataset, seed)
    
    train_loader = DataLoader(ContrastiveDataset(train_places,
                                                 normalize=params["train"]["normalize"]),
                              batch_size=params["train"]["batch_size"], shuffle=True, num_workers=8)
    
    valid_loader = DataLoader(ContrastiveDataset(valid_places,
                                                 normalize=params["train"]["normalize"]),
                              batch_size=params["train"]["batch_size"], shuffle=False, num_workers=8)
    
    return train_loader, valid_loader, train_places, valid_places


# --- private
def _load_and_split(train_dataset, val_dataset, seed):
    train_places = extract_places(train_dataset)
    valid_places = extract_places(val_dataset)
 
    if train_dataset == val_dataset:
        indices = np.arange(len(train_places))
        train_idx, valid_idx = train_test_split(indices, test_size=0.25, random_state=seed, shuffle=True)
        train_places = [train_places[i] for i in train_idx]
        valid_places = [valid_places[i] for i in valid_idx]
    
    return train_places, valid_places


class ContrastiveDataset(BaseDataset):
    """
    Restituisce tutte le immagini dello stesso place come positives.
    Per ora non gestisce day/night.
    """
    def __getitem__(self, index):
        place_images = self.paths[index]
        # carica tutte le immagini dello stesso place
        imgs = []
        crop = None
        for p in place_images:
            img, c = self.__load__(p)
            if crop is None:
                crop = c
            imgs.append(img)
        return torch.stack(imgs)  # batch di immagini positive

