import torch
import torch.nn as nn

class DinoV2Backbone(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', output_type='patch'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.output_type = output_type
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        
        features_dict = self.model.forward_features(x)
        
        if self.output_type == 'cls':
            return features_dict['x_norm_clstoken']
        
        x_patch = features_dict['x_norm_patchtokens']
        
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        
        x_patch = x_patch.permute(0, 2, 1)
        x_patch = x_patch.reshape(B, self.embed_dim, h_grid, w_grid)
        
        return x_patch

def build_dinov2(state_dict=None, model_type="vitb14"):
    repo_name = f"dinov2_{model_type}"
    
    backbone = DinoV2Backbone(model_name=repo_name, output_type='patch')
    
    if state_dict is not None:
        backbone.load_state_dict(torch.load(state_dict))
        
    output_dim = backbone.embed_dim
    
    return backbone, output_dim