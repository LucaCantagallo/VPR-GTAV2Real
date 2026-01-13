import math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional, ColorJitter

# --------------------------
# FUNZIONI PRIVATE (Mattoncini)
# --------------------------

def _load_image_as_tensor(path):
    image = Image.open(path).convert("RGB")
    return functional.to_tensor(image)

def _get_random_crop_coords(image_tensor, crop_size):
    _, h, w = image_tensor.shape
    th, tw = crop_size
    h_limit = max(h - th, 0)
    w_limit = max(w - tw, 0)
    h_start = np.random.randint(0, h_limit + 1) if h_limit > 0 else 0
    w_start = np.random.randint(0, w_limit + 1) if w_limit > 0 else 0
    return w_start, w_start + tw, h_start, h_start + th

def _apply_crop(image_tensor, coords):
    w_start, w_stop, h_start, h_stop = coords
    return image_tensor[:, h_start:h_stop, w_start:w_stop]

def _center_crop(image_tensor):
    min_side = min(image_tensor.shape[1], image_tensor.shape[2])
    return functional.center_crop(image_tensor, [min_side, min_side])

def _resize(image_tensor, height, width):
    return functional.resize(image_tensor, (height, width))

def _normalize(image_tensor, mean, std):
    return functional.normalize(image_tensor, mean, std)

def _random_grayscale(image_tensor, p):
    if np.random.random() < p:
        return functional.rgb_to_grayscale(image_tensor, num_output_channels=3)
    return image_tensor

def _random_horizontal_flip(image_tensor, p):
    if np.random.random() < p:
        return functional.hflip(image_tensor)
    return image_tensor

def _random_erasing(img_tensor, p, scale, ratio, value=0):
    if np.random.random() < p:
        c, h, w = img_tensor.shape
        area = h * w
        
        target_area = np.random.uniform(scale[0], scale[1]) * area
        aspect_ratio = np.random.uniform(ratio[0], ratio[1])
        
        h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if w_erase < w and h_erase < h:
            i = np.random.randint(0, h - h_erase + 1)
            j = np.random.randint(0, w - w_erase + 1)
            
            v = torch.ones((c, h_erase, w_erase)) * value
            img_tensor[:, i:i+h_erase, j:j+w_erase] = v
            
    return img_tensor

# --- NUOVA LOGICA EXTENDED (Zoom Out + Padding) ---
def _get_random_resized_crop_params_extended(img_tensor, scale, ratio):
    c, h, w = img_tensor.shape
    area = h * w
    
    # Target area (se scale > 1.0, stiamo facendo zoom out)
    target_area = np.random.uniform(scale[0], scale[1]) * area
    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
    aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
    
    w_crop = int(round(math.sqrt(target_area * aspect_ratio)))
    h_crop = int(round(math.sqrt(target_area / aspect_ratio)))
    
    # Calcolo Padding se il crop esce dall'immagine
    pad_h = max(0, h_crop - h)
    pad_w = max(0, w_crop - w)
    
    if pad_h > 0 or pad_w > 0:
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        # (left, top, right, bottom)
        padding = (pad_left, pad_top, pad_w - pad_left, pad_h - pad_top)
    else:
        padding = None

    eff_h = h + (pad_h if padding else 0)
    eff_w = w + (pad_w if padding else 0)
    
    i_limit = max(0, eff_h - h_crop)
    j_limit = max(0, eff_w - w_crop)
    
    i = np.random.randint(0, i_limit + 1)
    j = np.random.randint(0, j_limit + 1)
    
    return i, j, h_crop, w_crop, padding

def _apply_color_jitter(image_tensor, brightness, contrast, saturation, hue):
    jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    return jitter(image_tensor)

# --------------------------
# ORCHESTRATORE
# --------------------------

def preprocess_data(path, params, previous_crop=None):
    img = _load_image_as_tensor(path)
    crop_used = None

    # 1. CROP (Aggiornato)
    if params.get("use_random_resized_crop", False):
        if previous_crop is None:
            # Scale default ampio per supportare zoom out
            scale = params.get("rrc_scale", [0.5, 1.5])
            ratio = params.get("rrc_ratio", [3./4., 4./3.])
            i, j, h, w, padding = _get_random_resized_crop_params_extended(img, scale, ratio)
            crop_used = (i, j, h, w, padding)
        else:
            crop_used = previous_crop
            i, j, h, w, padding = crop_used
        
        if padding:
            img = functional.pad(img, padding, fill=0, padding_mode='constant')
        
        img = functional.crop(img, i, j, h, w)

    elif params.get("use_center_crop", False):
        img = _center_crop(img)
    elif params.get("use_random_crop", False):
        if previous_crop is None:
            side = min(img.shape[1], img.shape[2])
            crop_used = _get_random_crop_coords(img, [side, side])
        else:
            crop_used = previous_crop
        img = _apply_crop(img, crop_used)

    # 2. COLOR & TEXTURE
    if params.get("use_color_jitter", False):
        img = _apply_color_jitter(img, 
                                  params.get("jitter_brightness", 0), 
                                  params.get("jitter_contrast", 0),
                                  params.get("jitter_saturation", 0),
                                  params.get("jitter_hue", 0))

    if params.get("use_horizontal_flip", False):
        img = _random_horizontal_flip(img, params.get("flip_p", 0.5))

    if params.get("use_random_grayscale", False):
        img = _random_grayscale(img, params.get("grayscale_p", 0.1))

    # 3. RESIZE FINALE
    target_h = params.get("target_height", 224)
    target_w = params.get("target_width", 224)
    img = _resize(img, target_h, target_w)

    # 4. ERASING (Dopo il resize per coerenza dimensionale)
    if params.get("use_random_erasing", False):
        p = params.get("erasing_p", 0.5)
        scale = params.get("erasing_scale", [0.02, 0.33])
        ratio = params.get("erasing_ratio", [0.3, 3.3])
        img = _random_erasing(img, p, scale, ratio, value=0)

    # 5. NORMALIZE
    if params.get("normalize", False):
        mean = params.get("mean", [0.485, 0.456, 0.406])
        std = params.get("std", [0.229, 0.224, 0.225])
        img = _normalize(img, mean, std)

    return img, crop_used