### test.py

import yaml
import os
from glob import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from triplet_loader import test_paired_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import BaseDataset
from models import MLPCosine
from utils import load_params
from settings import get_device, get_test_work_dir, set_seed, init_model

class TestDataset(BaseDataset):
    def __init__(self, pairs, params):
        super().__init__(paths=pairs, params=params)

    def __getitem__(self, index):
        path_pair = self.paths[index]  
        img0, _ = self.__load__(path_pair[0])
        img1, _ = self.__load__(path_pair[1])
        return img0, img1

def compute_cm(features0, features1, work_dir, model_name):
    cm = np.zeros((len(features0), len(features1)))
    for i in range(len(features0)):
        cosine_sim = F.cosine_similarity(features0[i].unsqueeze(0), features1)
        cm[i] = cosine_sim
    np.savetxt(os.path.join(work_dir, f"cm_{model_name}.txt"), cm)

if __name__ == "__main__":
    device = get_device()
    
    experiments_dir = "./experiments"
    config_file = "./pipeline.yaml"
    params = load_params(config_file)

    work_dir = get_test_work_dir(params, experiments_dir=experiments_dir)
    
    set_seed(params["seed"])

    dataload_mode = params["dataload"]
    test_dataset = params["test_dataset"]

    test_places = test_paired_loader(dataload_mode, test_dataset)

    test_config = params["test"].get("preprocessing", {})

    dataset = TestDataset(test_places, params=test_config)
    
    dataloader = DataLoader(dataset, batch_size=params["train"]["batch_size"], shuffle=False, drop_last=False, pin_memory=True, num_workers=8, persistent_workers=False)

    model = init_model(params, device)
    model.eval()

    model_paths = glob(os.path.join(work_dir, "*.pt"))
    model_names = [os.path.split(path)[-1].split(".")[0] for path in model_paths]

    for j, path in enumerate(model_paths):
        model = MLPCosine.load_model_safely(model, path, device=device)
        model.to(device, non_blocking=True)
        model.eval()

        features_list = []
        labels_list = []

        for imgs0, imgs1 in tqdm(dataloader):
            f0 = model(imgs0.to(device))
            f1 = model(imgs1.to(device))
            features_list.append(f0.detach().cpu())
            labels_list.append(f1.detach().cpu())

        features_list = torch.cat(features_list)
        labels_list = torch.cat(labels_list)

        compute_cm(features_list, labels_list, work_dir, model_names[j])