import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional

# --------------------------
# FUNZIONI PRIVATE (Mattoncini)
# --------------------------

def _load_image_as_tensor(path):
    """Carica immagine da disco e converte in Tensor [C, H, W]"""
    image = Image.open(path).convert("RGB")
    return functional.to_tensor(image)

def _get_random_crop_coords(image_tensor, crop_size):
    """Calcola coordinate random per il crop (Logica originale)"""
    _, h, w = image_tensor.shape
    th, tw = crop_size
    h_limit = max(h - th, 0)
    w_limit = max(w - tw, 0)
    
    h_start = np.random.randint(0, h_limit + 1) if h_limit > 0 else 0
    w_start = np.random.randint(0, w_limit + 1) if w_limit > 0 else 0
    
    return w_start, w_start + tw, h_start, h_start + th

def _apply_crop(image_tensor, coords):
    """Applica il ritaglio usando le coordinate"""
    w_start, w_stop, h_start, h_stop = coords
    return image_tensor[:, h_start:h_stop, w_start:w_stop]

def _center_crop(image_tensor):
    """Applica crop centrale sul lato minore"""
    min_side = min(image_tensor.shape[1], image_tensor.shape[2])
    return functional.center_crop(image_tensor, [min_side, min_side])

def _resize(image_tensor, height, width):
    """Ridimensiona il tensore"""
    return functional.resize(image_tensor, (height, width))

def _normalize(image_tensor, mean, std):
    """Normalizza con media e deviazione standard"""
    return functional.normalize(image_tensor, mean, std)

def _random_grayscale(image_tensor, p):
    if np.random.random() < p:
        return functional.rgb_to_grayscale(image_tensor, num_output_channels=3)
    return image_tensor

def _random_gaussian_blur(image_tensor, p, kernel_size, sigma_range):
    if np.random.random() < p:
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        return functional.gaussian_blur(image_tensor, kernel_size=kernel_size, sigma=[sigma, sigma])
    return image_tensor

def _random_horizontal_flip(image_tensor, p):
    if np.random.random() < p:
        return functional.hflip(image_tensor)
    return image_tensor

def _get_random_resized_crop_params(img_tensor, scale, ratio):
    """Calcola i parametri i, j, h, w per il Random Resized Crop"""
    c, h, w = img_tensor.shape
    area = h * w
    
    target_area = np.random.uniform(scale[0], scale[1]) * area
    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
    aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
    
    w_crop = int(round(np.sqrt(target_area * aspect_ratio)))
    h_crop = int(round(np.sqrt(target_area / aspect_ratio)))
    
    if np.random.random() < 0.5:
        w_crop, h_crop = h_crop, w_crop
        
    if h_crop <= h and w_crop <= w:
        i = np.random.randint(0, h - h_crop + 1)
        j = np.random.randint(0, w - w_crop + 1)
        return i, j, h_crop, w_crop
    else:
        # Fallback: Center Crop se il calcolo fallisce (img troppo piccola)
        in_ratio = float(w) / float(h)
        if (in_ratio < min(ratio)):
            w_crop = w
            h_crop = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h_crop = h
            w_crop = int(round(h * max(ratio)))
        else:  
            w_crop = w
            h_crop = h
        i = (h - h_crop) // 2
        j = (w - w_crop) // 2
        return i, j, h_crop, w_crop
# --------------------------
# FUNZIONE PUBBLICA (Orchestratore)
# --------------------------

def preprocess_data(path, params, previous_crop=None):
    """
    Funzione unica di ingresso.
    Riceve il path e la configurazione (che simula lo yaml).
    Restituisce il tensore pronto e il crop usato.
    """
    
    # 1. Caricamento base
    img = _load_image_as_tensor(path)
    crop_used = None

    # 2. Gestione Logica Crop
    # Usa config.get() per replicare la lettura dei parametri
    if params.get("use_random_resized_crop", False):
            if previous_crop is None:
                scale = params.get("rrc_scale", [0.08, 1.0])
                ratio = params.get("rrc_ratio", [3./4., 4./3.])
                i, j, h, w = _get_random_resized_crop_params(img, scale, ratio)
                crop_used = (i, j, h, w)
            else:
                crop_used = previous_crop
                i, j, h, w = crop_used
            
            img = functional.crop(img, i, j, h, w)

    elif params.get("use_center_crop", False):
        img = _center_crop(img)
    
    elif params.get("use_random_crop", False):
        if previous_crop is None:
            # Calcola nuovo crop
            side = min(img.shape[1], img.shape[2])
            crop_used = _get_random_crop_coords(img, [side, side])
        else:
            # Usa crop precedente (per coerenza tra coppie se necessario)
            crop_used = previous_crop
        
        img = _apply_crop(img, crop_used)

    if params.get("use_horizontal_flip", False):
        img = _random_horizontal_flip(img, params.get("flip_p", 0.5))

    if params.get("use_random_grayscale", False):
        img = _random_grayscale(img, params.get("grayscale_p", 0.1))

    if params.get("use_gaussian_blur", False):
            kernel_size = params.get("blur_kernel_size", [5, 5])
            sigma_range = params.get("blur_sigma", [0.1, 2.0])
            p = params.get("blur_p", 0.5)
            img = _random_gaussian_blur(img, p, kernel_size, sigma_range)        
    # 3. Resize
    target_h = params.get("target_height", 224)
    target_w = params.get("target_width", 224)
    img = _resize(img, target_h, target_w)

    # 4. Normalize
    if params.get("normalize", False):
        mean = params.get("mean", [0.485, 0.456, 0.406])
        std = params.get("std", [0.229, 0.224, 0.225])
        img = _normalize(img, mean, std)

    return img, crop_used