# filter_loader_vpr.py
import numpy as np

def filter_paired_vpr(dataset_name, places):
    dataset_name = dataset_name.lower()
    

    pairs = []

    if dataset_name == "gta":
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs

    elif dataset_name == "alderley":
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs

    elif dataset_name == "tokyo247":
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs
    
    elif dataset_name == "gsv":
        for place in places:
            if len(place) < 2:
                continue
            idx = np.random.choice(len(place), 2, replace=False)
            pairs.append([place[idx[0]], place[idx[1]]])
        return pairs

    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
