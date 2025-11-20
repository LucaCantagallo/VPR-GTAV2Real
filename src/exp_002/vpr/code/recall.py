###recall.py

from glob import glob
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_n_folders, load_params



if __name__ == "__main__":    
    experiments_dir = "./experiments" 
    config_file = "./pipeline.yaml"
    params = load_params(config_file)
    
    base_path = os.path.join(experiments_dir, params["save_dir"])
    if params["experiment"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    elif str(params["experiment"]) == "-1":
        # Prende l'ultimo elemento numerico dentro base_path (ordinamento numerico corretto)
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
        work_dir = os.path.join(base_path, params["experiment"])
        
    root = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Tokyo247/Tokyo_24_7"
    places = glob(os.path.join(root, "*"))
    places_names = [os.path.split(p)[-1].split(".")[0] for p in places]
    
    matrix_paths = glob(os.path.join(work_dir, "cm_*.txt"))  
    k = params["evaluation"].get("k")
    
    for path in matrix_paths:
        matrix = np.loadtxt(path)
        name = os.path.split(path)[-1].split(".")[0]
        res = np.argsort(matrix, axis=1)
        to_print = ""
        recall = []
        for j in range(1, k+1):
            accuracy = 0
            for i in range(len(matrix)):              
                top_k = res[i, -j:]
                if i in top_k:
                    accuracy += 1
        
            accuracy /= len(matrix) 
            recall.append(accuracy)
            
        df = pd.DataFrame(recall)
        df.index += 1
        df.to_csv(os.path.join(work_dir, f"recall_{name}.txt"), sep=" ", header=None)

         
        
                