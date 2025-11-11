###dataset. py

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional
from itertools import permutations, combinations

def get_triplets(paths, samples_per_place=-1, use_combination=False):
    triplets = []
    indices = []
    for i, path in enumerate(paths): 
        idx = np.arange(0, len(path))
        if use_combination:
            pair = list(map(list, combinations(idx, 2)))
        else:
            pair = list(map(list, permutations(idx, 2)))
        indices.append(pair)

    for i, idx in enumerate(indices):
        if samples_per_place > 0 and len(idx) > samples_per_place:
            positive_indices = np.random.choice(len(idx), samples_per_place, False)
            current_triplets = np.array(paths[i])[idx][positive_indices].tolist()
        else:
            current_triplets = np.array(paths[i])[idx].tolist()           
        for j, elem in enumerate(current_triplets):
            while True:
                negative_idx = np.random.randint(0, len(paths))
                if negative_idx != i:
                    if len(idx) == 2:
                        negative_sample_idx = idx[j][1]
                    else:
                        negative_sample_idx = np.random.randint(len(paths[negative_idx]))
                    elem.append(str(paths[negative_idx][negative_sample_idx]))
                    break                
        triplets.extend(current_triplets)  
    return triplets

def get_random_crop(image, crop_size):
    c, h, w = image.shape
    th, tw = crop_size
    h_limit = h - th
    w_limit = w - tw
    h_start = 0
    w_start = 0
    h_stop = th if h > th else h
    w_stop = tw if w > tw else w
    if h_limit > 0:
        h_start = np.random.randint(0, h_limit)
    if w_limit > 0:
        w_start = np.random.randint(0, w_limit)
    return w_start, w_stop, h_start, h_stop
    
def crop_random(image, crop):
    w_start, w_stop, h_start, h_stop = crop
    return image[:, h_start:h_start + h_stop, w_start:w_start + w_stop]
    
class BaseDataset(Dataset):
    def __init__(self, 
                 paths,
                 target_width:int = 224, 
                 target_height:int = 224,
                 use_center_crop:bool = False,
                 use_random_crop:bool = False,
                 normalize:bool = False,
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]):
        self.paths = np.array(paths)
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
        # string_path = path.decode('utf-8')
        string_path = path
        image = Image.open(path)
        image_tensor = functional.to_tensor(image)
        if self.use_center_crop:
            if "alderley" in string_path:
                image_tensor = functional.center_crop(image_tensor, [346, 260])
            elif "GTAV" in string_path:
                image_tensor = functional.center_crop(image_tensor, [298, 224]) 
        if self.use_random_crop:
            if crop is None:
                if "alderley" in string_path:
                    crop = get_random_crop(image_tensor, [346, 260]) 
                elif "GTAV" in string_path:
                    crop = get_random_crop(image_tensor, [298, 224])
            image_tensor = crop_random(image_tensor, crop)
        image_tensor = functional.resize(image_tensor, (self.target_height, self.target_width))
        if self.normalize:
            image_tensor = functional.normalize(image_tensor, self.mean, self.std)
        return image_tensor, crop     
    
class TriCombinationDataset(BaseDataset):
    def __init__(self, 
                 paths,
                 target_width:int = 224, 
                 target_height:int = 224,
                 use_center_crop:bool = False,
                 use_random_crop:bool = False,
                 normalize:bool = False,
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]):
        super().__init__(paths, target_width, target_height, use_center_crop, use_random_crop, normalize, mean, std)

    
    def __getitem__(self, index):
        paths = self.paths[index]
       
        anchor, crop = self.__load__(paths[0])       
        positive, _ = self.__load__(paths[1], crop)
        negative, _ = self.__load__(paths[2])
        return anchor, positive, negative
    
class TestDataset(BaseDataset):
    def __init__(self, 
                 paths,
                 target_width:int = 224, 
                 target_height:int = 224,
                 use_center_crop:bool = False,
                 use_random_crop:bool = False,
                 normalize:bool = False,
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]):
        super().__init__(paths, target_width, target_height, use_center_crop, use_random_crop, normalize, mean, std)

    def __getitem__(self, index):
        return self.__load__(self.paths[index])[0]