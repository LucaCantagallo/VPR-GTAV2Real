import torch
from torch import nn
from collections import OrderedDict

def _get_named_children_until(model, layer_name):
    layers = OrderedDict()
    for name, child in model.named_children():
        if name == layer_name:
            break
        layers[name] = child
    return layers

def build_resnest50(state_dict=None):
    print("[INFO] Building ResNeSt-50 (Split-Attention)...")
    
    try:
        full_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    except Exception as e:
        print(f"[ERROR] Impossibile scaricare ResNeSt da torch.hub: {e}")
        raise e

    if state_dict is not None:
        full_model.fc = nn.Linear(2048, 1000) 
        full_model.load_state_dict(torch.load(state_dict))
    
    layers = _get_named_children_until(full_model, "avgpool")
    
    backbone = nn.Sequential(layers)
    output_dim = 2048
    
    return backbone, output_dim