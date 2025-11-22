## data_loader.py

from data_loader_vpr import load_paired_vpr
from data_loader_daynight import load_paired_daynight
    

def dataload_paired(mode, dataset_name):
    mode = mode.lower()

    if mode == "vpr":
        return load_paired_vpr(dataset_name)

    if mode == "daynight":
        return load_paired_daynight(dataset_name)

    raise ValueError(f"Dataload non gestito: {mode}")
