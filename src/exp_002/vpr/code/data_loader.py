## data_loader.py

import os
from glob import glob
import numpy as np
from utils import get_gta_places

def load_dataset(dataset_name):
    """
    Restituisce una lista di 'places'.
    - GTA: lista di coppie [day_img, night_img] (come in v1: get_gta_places mantiene la logica d_s / n_c).
    - Alderley: lista di coppie [img_i, img_i+1] (ordinato, pairing 0-1, 2-3, ...).
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "gta":
        root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GTAV"
        folders = glob(os.path.join(root, "*"))
        # per ciascuna cartella prendi i .jpg
        places = [glob(os.path.join(p, "*.jpg")) for p in folders]
        places = [p for p in places if len(p) > 0]

        # usa la stessa logica get_gta_places della v1
        day_pre = get_gta_places(places, "d_s")
        night_pre = get_gta_places(places, "n_c")

        day_places = []
        night_places = []
        for i, e in enumerate(night_pre):
            if len(e) > 0:
                day_places.append(day_pre[i])
                night_places.append(e)

        # costruisci la lista finale di "places" come [day_sample, night_sample]
        gta_paths = [[day_places[i][0], night_places[i][np.random.randint(0, len(night_places[i]))]]
                     for i in range(len(day_places))]
        return gta_paths

    elif dataset_name == "alderley":
        root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Alderley/alderley_paired"
        files = sorted(glob(os.path.join(root, "*.png")))
        # pairing 0–1, 2–3, ...
        places = [[files[i], files[i+1]] for i in range(0, len(files) - (len(files) % 2), 2)]
        return places

    elif dataset_name == "tokyo247":
        root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Tokyo247/Tokyo_24_7"
        files = sorted(glob(os.path.join(root, "*")))

        # costruisci le coppie [day, night] ignorando la sera (y)
        places = [[files[i], files[i+2]] for i in range(0, len(files) - 2, 3)]
        return places


    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
