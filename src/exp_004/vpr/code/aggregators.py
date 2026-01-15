import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
    
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        
    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1)).flatten(1)

def get_aggregator(agg_name, **kwargs):
    if agg_name == "gem":
        return GeM(**kwargs)
    elif agg_name == "avg":
        return AvgPool()
    elif agg_name == "max":
        return MaxPool()
    else:
        raise ValueError(f"Aggregator {agg_name} not implemented")