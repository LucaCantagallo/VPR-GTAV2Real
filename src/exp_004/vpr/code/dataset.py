import numpy as np
from torch.utils.data import Dataset
from preprocessing import preprocess_data


class BaseDataset(Dataset):
    def __init__(self, paths, params): 
        self.paths = np.array(paths)
        self.params = params 

    def __len__(self):
        return len(self.paths)

    def __load__(self, path, crop=None):
        image_tensor, crop_used = preprocess_data(path, self.params, previous_crop=crop)
        
        return image_tensor, crop_used