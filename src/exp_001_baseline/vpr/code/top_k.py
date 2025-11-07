#top_k.py

from glob import glob
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_n_folders, load_params



if __name__ == "__main__":    
    experiments_dir = "./experiments"
    
    config_file = "./top_k.yaml"
    params = load_params(config_file)
    
    base_path = os.path.join(experiments_dir, params["dataset"])
    if params["work_dir"] is None:
        n_folders = get_n_folders(base_path)
        work_dir = os.path.join(base_path, str(n_folders))
    else:
        work_dir = os.path.join(base_path, params["work_dir"])
        
    root = "/home/lcantagallo/VPR-GTAV2Real/src/exp_001_baseline/vpr/dataset/Tokyo247/Tokyo_24_7" #TODO change to the correct path
    places = glob(os.path.join(root, "*"))
    places_names = [os.path.split(p)[-1].split(".")[0] for p in places]
    
    matrix_paths = glob(os.path.join(work_dir, "cm_*.txt"))
    
    k = params["k"]
    
    for path in matrix_paths:
        accuracy = 0
        matrix = np.loadtxt(path)
        name = os.path.split(path)[-1].split(".")[0]
        res = np.argsort(matrix, axis=1)
        to_print = ""
        for i in range(len(matrix)):
            top_k = res[i, -k:]
            
            to_print += f"{places_names[i]}: "
            for e in top_k:
                to_print += places_names[e] + " "
            to_print += "\n"
            if i in top_k:
                accuracy += 1
                    
        accuracy /= len(matrix) 
        with open(os.path.join(work_dir, f"top_{k}_{name}.txt"), "a") as f:
            f.write(str(accuracy) + "\n" + to_print)
         
        
                