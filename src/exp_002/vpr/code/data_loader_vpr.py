# data_loader_vpr.py
from places_extractor import extract_places
import numpy as np

def load_vpr(dataset_name):
    dataset_name = dataset_name.lower()
    

    pairs = []

    if dataset_name == "gta":
        places = extract_places(dataset_name)
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs

    elif dataset_name == "alderley":
        places = extract_places(dataset_name)
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs

    elif dataset_name == "tokyo247":
        places = extract_places(dataset_name)
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs
    
    elif dataset_name == "gsv":
        places = extract_places(dataset_name, percentage=0.05)
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs

    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
