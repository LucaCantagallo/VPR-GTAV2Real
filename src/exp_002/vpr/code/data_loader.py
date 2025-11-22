from data_loader_vpr import load_vpr
from data_loader_daynight import load_daynight

def dataload(mode, dataset_name):
    mode = mode.lower()

    if mode == "vpr":
        return load_vpr(dataset_name)

    if mode == "daynight":
        return load_daynight(dataset_name)

    raise ValueError(f"Dataload non gestito: {mode}")
