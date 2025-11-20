# places_extractor.py
import os
from glob import glob

def extract_places(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "gta":
        root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GTAV"
        folders = glob(os.path.join(root, "*"))
        places = [glob(os.path.join(p, "*.jpg")) for p in folders if len(glob(os.path.join(p, "*.jpg"))) > 0]
        return places

    elif dataset_name == "alderley":
        root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Alderley/alderley_paired"
        files = sorted(glob(os.path.join(root, "*.png")))
        places = [[files[i], files[i+1]] for i in range(0, len(files) - (len(files) % 2), 2)]
        return places

    elif dataset_name == "tokyo247":
        root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Tokyo247/Tokyo_24_7"
        files = sorted(glob(os.path.join(root, "*")))
        places = [files[i:i+3] for i in range(0, len(files) - 2, 3)]
        return places

    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
