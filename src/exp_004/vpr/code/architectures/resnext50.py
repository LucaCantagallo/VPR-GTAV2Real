import torch
from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from collections import OrderedDict

def _get_named_children_until(model, layer_name):
    layers = OrderedDict()
    for name, child in model.named_children():
        if name != layer_name:
            layers[name] = child
        else:
            break
    return layers

def build_resnext50(state_dict=None):
    print("[INFO] Building ResNeXt-50 (32x4d)...")
    
    if state_dict is None:
        full_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    else:
        full_model = resnext50_32x4d()
        full_model.fc = nn.Linear(2048, 365) 
        full_model.load_state_dict(torch.load(state_dict))
    
    layers = _get_named_children_until(full_model, "fc")
    layers["flatten"] = nn.Flatten()
    
    backbone = nn.Sequential(layers)
    
    output_dim = 2048
    
    return backbone, output_dim