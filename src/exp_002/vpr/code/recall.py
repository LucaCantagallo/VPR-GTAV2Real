###recall.py

from glob import glob
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from settings import get_test_work_dir
from utils import get_n_folders, load_params



if __name__ == "__main__":    
    experiments_dir = "./experiments" 
    config_file = "./pipeline.yaml"
    params = load_params(config_file)
    
    work_dir = get_test_work_dir(params, experiments_dir=experiments_dir)
    
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

         
        
                