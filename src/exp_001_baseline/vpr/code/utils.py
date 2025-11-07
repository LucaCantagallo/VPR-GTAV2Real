##utils.py

import yaml
import os

def load_params(config_file):
    with open(config_file) as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
def save_params(file):
    with open(file) as f:
        try:
            return yaml.safe_dump(f)
        except yaml.YAMLError as exc:
            print(exc)
            
def get_n_folders(root):
    return len(next(os.walk(root))[1])

def unroll(l):
    return [e for j in l for e in j]

def get_gta_places(paths, weather):
    return [[l for l in p if weather in l] for p in paths]


