# data_loader_daynight.py
import numpy as np

def filter_paired_daynight(dataset_name, places):
    dataset_name = dataset_name.lower()
    

    if dataset_name == "gta":
        day_pre = _get_gta_places(places, "d_s")
        night_pre = _get_gta_places(places, "n_c")

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
        return places

    elif dataset_name == "tokyo247":
        return [[p[0], p[2]] for p in places]

    else:
        raise ValueError(f"Dataset non gestito: {dataset_name}")
    
def _get_gta_places(paths, weather):
    return [[l for l in p if weather in l] for p in paths]
