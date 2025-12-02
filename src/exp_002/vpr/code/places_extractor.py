### places_extractor.py

import yaml
from glob import glob
import os
import random

with open("dataset_path.yaml") as f:
    DATASET_ROOTS = yaml.safe_load(f)["datasets"]

def extract_places(dataset_name, percentage=1.0, seed=42):
    dataset_name = dataset_name.lower()
    places = []

    if dataset_name == "gta":
        root = DATASET_ROOTS[dataset_name]
        places = _places_extractor_gta(root)

    elif dataset_name == "alderley":
        root = DATASET_ROOTS[dataset_name]
        places = _places_extractor_alderley(root)

    elif dataset_name == "tokyo247":
        root = DATASET_ROOTS[dataset_name]
        places = _places_extractor_tokyo247(root)

    elif dataset_name == "gsv":
        root = DATASET_ROOTS[dataset_name]
        places = _places_extractor_gsv(root)
        percentage = 0.005

    elif dataset_name == "gsv_valid":
        root = DATASET_ROOTS["gsv"]
        places = _places_extractor_gsv(root)[0::2]
        percentage = 0.01

    elif dataset_name == "gsv_test":
        root = DATASET_ROOTS["gsv"]
        places = _places_extractor_gsv(root)[1::2]
        percentage = 0.01

    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")

    if percentage < 1.0:
        subset_size = max(1, int(len(places) * percentage))
        random.seed(seed)
        places = random.sample(places, subset_size)

    return places



def _places_extractor_gta(root):
    folders = glob(os.path.join(root, "*"))
    return [
        glob(os.path.join(p, "*.jpg"))
        for p in folders
        if len(glob(os.path.join(p, "*.jpg"))) > 0
    ]


def _places_extractor_alderley(root):
    files = sorted(glob(os.path.join(root, "*.png")))
    return [
        [files[i], files[i+1]]
        for i in range(0, len(files) - (len(files) % 2), 2)
    ]


def _places_extractor_tokyo247(root):
    files = sorted(glob(os.path.join(root, "*")))
    return [
        files[i:i+3]
        for i in range(0, len(files) - 2, 3)
    ]


def _places_extractor_gsv(root):
    city_folders = glob(os.path.join(root, "Images", "*"))
    places = []
    for city in city_folders:
        files = glob(os.path.join(city, "*.jpg"))
        groups = {}
        for f in files:
            fname = os.path.basename(f)
            place_id = "_".join(fname.split("_")[:2])
            groups.setdefault(place_id, []).append(f)

        for imgs in groups.values():
            if len(imgs) >= 4:
                places.append(imgs)

    return places

