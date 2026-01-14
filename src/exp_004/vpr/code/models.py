import torch
from torch import nn
from collections import OrderedDict
import os

from architectures.resnet50 import build_resnet50 
from architectures.resnext50 import build_resnext50 
from architectures.resnest50 import build_resnest50

class BaseFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_name: str = "resnet50",
                 state_dict = None,
                 trainable_from_layer: str = None,
                 device: str = "cpu",
                 **kwargs):
        super(BaseFeatureExtractor, self).__init__()

        print(f"\n{'='*30}")
        print(f"[MODEL CHECK] Richiesto modello: '{model_name}'")
        
        if model_name == "resnet50":
            self.model_base, self.output_dim = build_resnet50(state_dict)
        elif model_name == "resnext50":
            self.model_base, self.output_dim = build_resnext50(state_dict)
        elif model_name == "resnest50": 
            self.model_base, self.output_dim = build_resnest50(state_dict)
        else:
            raise ValueError(f"Modello '{model_name}' non supportato o non trovato in architectures.")
        

        self.freezed_part = OrderedDict()
        self.trainable_part = OrderedDict()

        if trainable_from_layer == "all":
            self.trainable_part = self.model_base
            self.freezed_part = None 
        elif trainable_from_layer is not None:
            found = False
            for name, child in self.model_base.named_children():
                if name == trainable_from_layer:
                    found = True
                if found:
                    self.trainable_part[name] = child
                else:
                    self.freezed_part[name] = child            
        else:
            self.freezed_part = self.model_base    
            self.trainable_part = None 
        
        if isinstance(self.freezed_part, OrderedDict):
            if len(self.freezed_part) > 0:
                self.freezed_part = nn.Sequential(self.freezed_part)
            else:
                self.freezed_part = None

        if isinstance(self.trainable_part, OrderedDict):
            if len(self.trainable_part) > 0:
                self.trainable_part = nn.Sequential(self.trainable_part)
            else:
                self.trainable_part = None

        if self.freezed_part is not None:
            self.freezed_part.to(device)
            for param in self.freezed_part.parameters():
                param.requires_grad = False
            self.freezed_part.eval() 
            
        if self.trainable_part is not None:
            self.trainable_part.to(device)
        
        self.to(device=device)
        
    def train(self, mode: bool = True):
        if self.freezed_part is not None:
            self.freezed_part.eval()
        if self.trainable_part is not None:
            self.trainable_part.train(mode)  
            
    def extract_features(self, x):
        if self.freezed_part is not None:
            with torch.no_grad():
                x = self.freezed_part(x)
        if self.trainable_part is not None:
            x = self.trainable_part(x)
        return x

class MLPCosine(BaseFeatureExtractor):
    def __init__(self, 
                 model_name: str = "resnet50",
                 state_dict: str = None,
                 trainable_from_layer: str = None,
                 device: str = "cpu",
                 **kwargs):
        
        super(MLPCosine, self).__init__(
            model_name=model_name, 
            state_dict=state_dict, 
            trainable_from_layer=trainable_from_layer, 
            device=device,
            **kwargs
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)
        ).to(device=device)
        
        self.to(device)
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.mlp(x)
        return x
    
    @staticmethod
    def load_model_safely(model, checkpoint_path, device="cpu"):
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("[INFO] No checkpoint provided — using current model weights.")
            return model

        print(f"[INFO] Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)

        model_keys = list(model.state_dict().keys())
        ckpt_keys = list(state_dict.keys())

        ckpt_has_trainable = any("trainable_part." in k for k in ckpt_keys)
        ckpt_has_freezed = any("freezed_part." in k for k in ckpt_keys)
        model_has_trainable = any("trainable_part." in k for k in model_keys)
        model_has_freezed = any("freezed_part." in k for k in model_keys)

        if (ckpt_has_trainable == model_has_trainable) and (ckpt_has_freezed == model_has_freezed):
            model.load_state_dict(state_dict, strict=True)
            print("[INFO] Weights loaded successfully (matching parts).")
            return model

        if ckpt_has_trainable and model_has_freezed:
            print("[WARN] Inverted checkpoint detected — remapping trainable/freezed keys.")
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("trainable_part."):
                    remapped[k.replace("trainable_part.", "freezed_part.")] = v
                elif k.startswith("freezed_part."):
                    remapped[k.replace("freezed_part.", "trainable_part.")] = v
                else:
                    remapped[k] = v
            model.load_state_dict(remapped, strict=True)
            print("[INFO] Weights loaded after remapping.")
            return model

        if not ckpt_has_trainable and not ckpt_has_freezed:
            print("[INFO] Loading base pretrained weights into matching layers.")
            model_dict = model.state_dict()
            common = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(common)
            model.load_state_dict(model_dict, strict=True)
            print(f"[INFO] Loaded {len(common)}/{len(model_dict)} layers from base checkpoint.")
            return model

        print("[WARN] Partial mismatch — loading overlapping keys only.")
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=True)
        print(f"[INFO] Loaded {len(matched)}/{len(model_dict)} compatible layers.")
        return model