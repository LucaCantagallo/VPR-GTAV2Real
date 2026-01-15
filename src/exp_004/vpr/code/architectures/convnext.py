import torch
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def build_convnext_tiny(state_dict=None):
    print("[INFO] Building ConvNeXt-Tiny...")
    
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    full_model = convnext_tiny(weights=weights)

    if state_dict is not None:
        full_model.classifier[2] = nn.Linear(768, 1000)
        full_model.load_state_dict(torch.load(state_dict))

    backbone = full_model.features
    
    output_dim = 768
    
    return backbone, output_dim