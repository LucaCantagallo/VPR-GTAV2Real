import numpy as np
from torch.utils.data import Dataset
from preprocessing import preprocess_data


class BaseDataset(Dataset):
    def __init__(self, 
                 paths,
                 target_width:int = 224, 
                 target_height:int = 224,
                 use_center_crop:bool = False,
                 use_random_crop:bool = False,
                 normalize:bool = False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        
        self.paths = np.array(paths)
        # Salviamo i parametri passati dagli altri script/yaml
        self.target_width = target_width
        self.target_height = target_height
        self.use_center_crop = use_center_crop
        self.use_random_crop = use_random_crop
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __load__(self, path, crop=None):
        # COSTRUZIONE DEL PONTE:
        # Creiamo un dizionario di configurazione usando i valori
        # che questa classe ha ricevuto. In questo modo preprocessing.py
        # lavora come se leggesse uno yaml, anche se i dati arrivano da qui.
        current_config = {
            "target_width": self.target_width,
            "target_height": self.target_height,
            "use_center_crop": self.use_center_crop,
            "use_random_crop": self.use_random_crop,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std
        }

        # Chiamata all'unica funzione pubblica esterna
        image_tensor, crop_used = preprocess_data(path, current_config, previous_crop=crop)
        
        return image_tensor, crop_used