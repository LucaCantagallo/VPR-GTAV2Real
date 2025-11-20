### places_extractor.py

import yaml
from glob import glob
import os
import random

with open("dataset_path.yaml") as f:
    DATASET_ROOTS = yaml.safe_load(f)["datasets"]

def extract_places(dataset_name, percentage=1.0, seed=42):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_ROOTS:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
    
    root = DATASET_ROOTS[dataset_name]
    places = []

    if dataset_name == "gta":
        folders = glob(os.path.join(root, "*"))
        places = [glob(os.path.join(p, "*.jpg")) for p in folders if len(glob(os.path.join(p, "*.jpg"))) > 0]

    elif dataset_name == "alderley":
        files = sorted(glob(os.path.join(root, "*.png")))
        places = [[files[i], files[i+1]] for i in range(0, len(files) - (len(files) % 2), 2)]

    elif dataset_name == "tokyo247":
        files = sorted(glob(os.path.join(root, "*")))
        places = [files[i:i+3] for i in range(0, len(files) - 2, 3)]
    
    elif dataset_name == "gsv":
    # Prende tutte le cartelle delle città
        city_folders = glob(os.path.join(root, "Images", "*"))
        

        for city in city_folders:
            files = glob(os.path.join(city, "*.jpg"))
            groups = {}
            for f in files:
                fname = os.path.basename(f)
                # Estraggo city + place_id (primi due token del nome)
                # es: "Bangkok_0000001"
                place_id = "_".join(fname.split("_")[:2])
                groups.setdefault(place_id, []).append(f)

            # filtro: includo solo places con >= 4 immagini
            for imgs in groups.values():
                if len(imgs) >= 4:
                    places.append(imgs)

    if(percentage < 1.0):
        subset_size = max(1, int(len(places) * percentage))
        random.seed(seed)
        places = random.sample(places, subset_size)

    return places
