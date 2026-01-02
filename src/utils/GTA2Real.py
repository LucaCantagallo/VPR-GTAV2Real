import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

INPUT_DIR = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GTAVSamples"
OUTPUT_DIR = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Output_Real_SD15"
MODEL_ID = "emilianJR/epiCRealism" 
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_canny"

def resize_maintain_aspect(image, max_size=768):
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        new_w = new_w - (new_w % 8)
        new_h = new_h - (new_h % 8)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image

def main():
    if not torch.cuda.is_available():
        print("ERRORE: GPU non trovata!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"ERRORE: La cartella di input non esiste: {INPUT_DIR}")
        return

    print(f"Caricamento modelli... (Target: {MODEL_ID})")
    
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID, 
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        print("Xformers non disponibile, procedo senza.")

    pipe.enable_model_cpu_offload() 
    
    print("Pipeline pronta.")

    PROMPT = "photo of a street, urban city, asphalt road, buildings, realistic lighting, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    NEGATIVE_PROMPT = "cartoon, anime, 3d, painting, b&w, draw, bad art, deform, blur, bad quality, lowres, watermark"

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Trovate {len(files)} immagini da convertire in: {INPUT_DIR}")

    for i, filename in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processando {filename}...")
        
        try:
            img_path = os.path.join(INPUT_DIR, filename)
            image = load_image(img_path)
            
            image = resize_maintain_aspect(image, max_size=768)
            image_np = np.array(image)

            low_threshold = 100
            high_threshold = 200
            canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            canny_pil = Image.fromarray(canny_image)

            result = pipe(
                PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=canny_pil,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.5,
            ).images[0]

            result.save(os.path.join(OUTPUT_DIR, filename))
            
        except Exception as e:
            print(f"Errore su {filename}: {e}")
            torch.cuda.empty_cache()

    print(f"Conversione completata. Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()