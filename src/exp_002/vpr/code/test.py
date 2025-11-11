###test.py

import yaml
import os
from glob import glob
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import TestDataset
from models import MLPCosine
from utils import load_params, get_n_folders

def test(features0, features1, cm, j, name):
    for i in range(len(features0)):
        cosine_sim = F.cosine_similarity(features0[i].unsqueeze(0), features1)
        cm[i] = cosine_sim
                    
    np.savetxt(os.path.join(work_dir, f"cm_{name}_{model_names[j]}.txt"), cm)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    experiments_dir = "./experiments"
    
    config_file = "./test.yaml"
    params = load_params(config_file)
    
    base_path = os.path.join(experiments_dir, params["dataset"])

    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))

    elif str(params["work_dir"]) == "-1":
        subfolders = [
            f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()
        ]
        if not subfolders:
            raise ValueError(f"Nessuna sottocartella trovata in {base_path}")
        
        # Ordina numericamente invece che alfabeticamente
        subfolders = sorted(subfolders, key=lambda x: int(x))
        work_dir = os.path.join(base_path, subfolders[-1])


    else:
        work_dir = os.path.join(base_path, str(params["work_dir"]))
            
        seed = params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)    
    
    root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Tokyo247/Tokyo_24_7" #TODO change to the correct path
    places = sorted(glob(os.path.join(root, "*")))
    
    places = [places[i] for i in range(len(places)) if i % 3 == 0 or i % 3 == 2]

    #print(places[:100])
    
    dataset = TestDataset(places)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True, num_workers=8, persistent_workers=False)
    
    model = MLPCosine()
    model.to(device=device, non_blocking=True)
    
    model_paths = glob(os.path.join(work_dir, "*.pt"))
    model_names = [os.path.split(path)[-1].split(".")[0] for path in model_paths]
        
    for j, path in enumerate(model_paths):
        model = MLPCosine.load_model_safely(model, path, device=device)
        #model.load_state_dict(torch.load(path))
        model.to(device, non_blocking=True)
        model.eval()
        
        features = []
    
        for mb in tqdm(dataloader):
            f = model(mb.to(device))
            features.append(f.detach().cpu())
            
        features = torch.cat(features)
        
        day_features = features[::2]
        night_features = features[1::2]
        
        cm_day = np.zeros((len(places) // 2, len(places) // 2))
        cm_night = np.zeros((len(places) // 2, len(places) // 2))
        
        test(day_features, night_features, cm_day, j, "day")
        test(night_features, day_features, cm_night, j, "night")   
    