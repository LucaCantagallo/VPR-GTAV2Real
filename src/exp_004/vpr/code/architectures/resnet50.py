import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict

def _get_named_children_until(model, layer_name):
    layers = OrderedDict()
    for name, child in model.named_children():
        if name != layer_name:
            layers[name] = child
        else:
            break
    return layers

def build_resnet50(state_dict=None):
    if state_dict is None:
        full_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        full_model = resnet50()
        full_model.fc = nn.Linear(2048, 365)
        full_model.load_state_dict(torch.load(state_dict))
    
    layers = _get_named_children_until(full_model, "fc")
    layers["flatten"] = nn.Flatten()
    
    backbone = nn.Sequential(layers)
    output_dim = 2048
    
    return backbone, output_dim