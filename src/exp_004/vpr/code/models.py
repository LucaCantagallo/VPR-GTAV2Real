import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import os

from architectures.resnet50 import build_resnet50 
from architectures.resnext50 import build_resnext50 
from architectures.resnest50 import build_resnest50
from architectures.convnext import build_convnext_tiny
from aggregators import get_aggregator

class VPRModel(nn.Module):
    def __init__(self, 
                 model_name: str = "resnet50",
                 aggregator: str = "avg",         
                 aggregator_params: dict = {}, 
                 output_dim: int = 2048,          
                 head_type: str = "linear",        
                 normalize_output: bool = False,   
                 state_dict = None,
                 trainable_from_layer: str = None,
                 device: str = "cpu",
                 **kwargs):
        super(VPRModel, self).__init__()

        print(f"\n{'='*30}")
        print(f"[MODEL CONFIG] Backbone: {model_name} | Aggregator: {aggregator} | Head: {head_type} | Norm: {normalize_output}")

        self.model_name = model_name

        if model_name == "resnet50":
            self.backbone, self.backbone_dim = build_resnet50(state_dict)
        elif model_name == "resnext50":
            self.backbone, self.backbone_dim = build_resnext50(state_dict)
        elif model_name == "resnest50": 
            self.backbone, self.backbone_dim = build_resnest50(state_dict)
        elif model_name == "convnext_tiny":  
            self.backbone, self.backbone_dim = build_convnext_tiny(state_dict)
        else:
            raise ValueError(f"Modello '{model_name}' non supportato.")
        
        num_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"[MODEL CHECK] Parametri Backbone: {num_params:,}")
        print(f"[MODEL CHECK] Backbone Output Channels: {self.backbone_dim}")

        self.aggregator_layer = get_aggregator(aggregator, **aggregator_params)
        
        self.head_type = head_type
        self.normalize_output = normalize_output
        
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim)
            )
        elif head_type == "linear":
            if output_dim and output_dim != self.backbone_dim:
                self.head = nn.Linear(self.backbone_dim, output_dim)
            else:
                self.head = nn.Identity()
        else:
             raise ValueError(f"Head type '{head_type}' non supportato (usa 'linear' o 'mlp')")

        self.freezed_part = OrderedDict()
        self.trainable_part = OrderedDict()

        if trainable_from_layer == "all":
            self.trainable_part = self.backbone
            self.freezed_part = None 
        elif trainable_from_layer is not None:
            found = False
            for name, child in self.backbone.named_children():
                if name == trainable_from_layer:
                    found = True
                if found:
                    self.trainable_part[name] = child
                else:
                    self.freezed_part[name] = child            
        else:
            self.freezed_part = self.backbone    
            self.trainable_part = None 
        
        if isinstance(self.freezed_part, OrderedDict) and len(self.freezed_part) > 0:
            self.freezed_part = nn.Sequential(self.freezed_part)
        else:
            if self.freezed_part is not None and len(self.freezed_part) == 0: self.freezed_part = None

        if isinstance(self.trainable_part, OrderedDict) and len(self.trainable_part) > 0:
             self.trainable_part = nn.Sequential(self.trainable_part)
        else:
             if self.trainable_part is not None and len(self.trainable_part) == 0: self.trainable_part = None

        if self.freezed_part:
            self.freezed_part.to(device)
            self.freezed_part.eval()
            for param in self.freezed_part.parameters():
                param.requires_grad = False
            
        if self.trainable_part:
            self.trainable_part.to(device)
            
        self.aggregator_layer.to(device)
        self.head.to(device)
        self.to(device=device)
        
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freezed_part:
            self.freezed_part.eval()
            
    def forward(self, x):
        if self.freezed_part:
            with torch.no_grad():
                x = self.freezed_part(x)
        if self.trainable_part:
            x = self.trainable_part(x)
        
        if isinstance(x, dict):
             x = x['x_norm_clstoken'] 

        x = self.aggregator_layer(x)
        
        if x.dim() > 2:
            x = x.flatten(1)
        
        x = self.head(x)

        if self.normalize_output:
            x = F.normalize(x, p=2, dim=1)
            
        return x
    
    @staticmethod
    def load_model_safely(model, checkpoint_path, device="cpu"):
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("[INFO] No checkpoint provided — using current model weights.")
            return model
        print(f"[INFO] Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        compatible = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(compatible) > 0:
             model.load_state_dict(compatible, strict=False)
             print(f"[INFO] Loaded {len(compatible)} layers.")
        else:
             print("[WARN] No compatible layers found!")
        return model