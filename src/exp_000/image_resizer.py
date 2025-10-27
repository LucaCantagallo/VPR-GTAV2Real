from glob import glob
import os
from PIL import Image
from tqdm import tqdm

def resize(path, target_size, target_path):
    image_name = os.path.split(path)[-1]
    image = Image.open(path)
    image = image.resize(target_size)
    new_path = os.path.join(target_path, image_name)
    image.save(new_path)
    
def resize_all(paths, target_size, target_path):
    for path in tqdm(paths):
        resize(path, target_size, target_path)


def resize_gta():
    gta_target_size = (398, 224)
    gta_target_dir = "/home/lcantagallo/VPR-GTAV2Real/dataset/GTAV_resized"
    
    if gta_target_dir is not None:
        os.makedirs(gta_target_dir, exist_ok=True)

    root = "/home/lcantagallo/VPR-GTAV2Real/dataset/GTAV"
    places = glob(os.path.join(root, "*"))
    
    for place in places:
        place_name = os.path.split(place)[-1]        
        target_path = os.path.join(gta_target_dir, place_name) 
        os.makedirs(target_path, exist_ok=True)
        images = glob(os.path.join(place, "*.jpg"))
        resize_all(images, gta_target_size, target_path)
  
if __name__ == "__main__":
    resize_gta()