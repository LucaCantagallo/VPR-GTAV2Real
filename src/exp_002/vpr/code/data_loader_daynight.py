# data_loader_daynight.py
from places_extractor import extract_places
from utils import get_gta_places
import numpy as np

def load_daynight(dataset_name):
    dataset_name = dataset_name.lower()
    

    if dataset_name == "gta":
        places = extract_places(dataset_name)
        day_pre = get_gta_places(places, "d_s")
        night_pre = get_gta_places(places, "n_c")

        day_places = []
        night_places = []
        for i, e in enumerate(night_pre):
            if len(e) > 0:
                day_places.append(day_pre[i])
                night_places.append(e)

        places = [[day_places[i][0], night_places[i][np.random.randint(0, len(night_places[i]))]]
                     for i in range(len(day_places))]
        return places

    elif dataset_name == "alderley":
        places = extract_places(dataset_name)
        return places

    elif dataset_name == "tokyo247":
        places = extract_places(dataset_name)
        return [[p[0], p[2]] for p in places]

    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
