import torch
from torch import nn
from collections import OrderedDict

def _get_named_children_until(model, layer_name):
    layers = OrderedDict()
    for name, child in model.named_children():
        if name != layer_name:
            layers[name] = child
        else:
            break
    return layers

def build_resnest50(state_dict=None):
    print("[INFO] Building ResNeSt-50 (Split-Attention)...")
    
    # Carichiamo dal repo ufficiale tramite torch.hub
    # Nota: La prima volta scaricherà i pesi nella cache
    try:
        full_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    except Exception as e:
        print(f"[ERROR] Impossibile scaricare ResNeSt da torch.hub: {e}")
        raise e

    if state_dict is not None:
        # Se hai pesi custom, dobbiamo ricreare la testa originale per caricarli
        # ResNeSt standard usa 'fc' come ultimo layer, esattamente come ResNet
        full_model.fc = nn.Linear(2048, 1000) 
        full_model.load_state_dict(torch.load(state_dict))
    
    # Tagliamo via il layer finale (fc) per avere solo il backbone
    # ResNeSt mantiene la struttura standard: layer1...layer4 -> avgpool -> fc
    layers = _get_named_children_until(full_model, "fc")
    
    # ResNeSt richiede il Flatten finale dopo l'avgpool
    layers["flatten"] = nn.Flatten()
    
    backbone = nn.Sequential(layers)
    
    # ResNeSt-50 esce a 2048 canali (come ResNet50)
    output_dim = 2048
    
    return backbone, output_dim