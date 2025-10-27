import torch
from torch import nn
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights

def _get_named_children_until(model, layer_name):
    layers = OrderedDict()
    for name, child in model.named_children():
        if name != layer_name:
            layers[name] = child
        else:
            break
    return layers

def _get_resnet(model):
    layers = _get_named_children_until(model, "fc")
    layers["flatten"] = nn.Flatten()
    return nn.Sequential(layers)

def _get_model(name, state_dict=None):
    print(state_dict)
    if name == "resnet50":
        if state_dict is None:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = resnet50()
            model.fc = nn.Linear(2048, 365)
            model.load_state_dict(torch.load(state_dict))
        return _get_resnet(model), 2048

class BaseFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_name: str = "resnet50",
                 state_dict = None,
                 trainable_from_layer: str = None,
                 device: str = "cpu",
                 **kwargs):
        super(BaseFeatureExtractor, self).__init__()
        model, self.output_dim = _get_model(model_name, state_dict)
        
        self.freezed_part = OrderedDict()
        self.trainable_part = OrderedDict()

        if trainable_from_layer == "all":
            self.trainable_part = model
        elif trainable_from_layer is not None:
            found = False
            
            for name, child in model.named_children():
                if name == trainable_from_layer:
                    found = True
                if found:
                    self.trainable_part[name] = child
                else:
                    self.freezed_part[name] = child            
        else:
            self.freezed_part = model    
            
        self.freezed_part = nn.Sequential(self.freezed_part).to(device, non_blocking=True) if len(self.freezed_part) > 0 else None    
        self.trainable_part = nn.Sequential(self.trainable_part).to(device, non_blocking=True) if len(self.trainable_part) > 0 else None

        assert self.trainable_part is not None or self.freezed_part is not None, "Feature extractor must have at least one model (freezed or trainable)"

        if self.freezed_part is not None:
            for param in self.freezed_part.parameters():
                param.requires_grad = False
            self.freezed_part.eval() 
        
        self.to(device=device, non_blocking=True)
        
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
                 trainable_from_layer: str = None,
                 device: str = "cpu",
                 **kwargs):
        super(MLPCosine, self).__init__(model_name, trainable_from_layer, device)
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim)).to(device=device, non_blocking=True)
        
        self.to(device, non_blocking=True)
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.mlp(x)
        return x
    
