#########loop.py

import torch
from tqdm import tqdm

def run_epoch(model,
         dataloader,
         loss_fn,
         optimizer=None,
         train=True,
         device="cpu"
         ):
    if train:
        model.train()
    else:
        model.eval()
    loss_val = 0.0
    
    with torch.set_grad_enabled(train):
        for mb_idx, mb in enumerate(tqdm(dataloader)):            
            
            anchor, positive, negative = mb
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor), model(positive), model(negative)
            loss = loss_fn(anchor_output, positive_output, negative_output) 

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            loss_val += loss.item()            
            
    loss_val /= (mb_idx + 1)
    return loss_val

        