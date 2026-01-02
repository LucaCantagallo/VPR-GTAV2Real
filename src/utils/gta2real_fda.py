import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms

# --- CONFIGURAZIONE ---
# Cartella con le tue immagini GTA
SRC_DIR = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GTAVSamples"

# Cartella con immagini REALI (usane una qualsiasi: Mapillary, Cityscapes, o GSV)
# Se non hai questa cartella popolata, metti qui dentro anche solo 5-10 foto reali.
TRG_DIR = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GSVSamples" 

OUTPUT_DIR = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Output_Real_FDA"
BETA = 0.05  # Quanto "spingere" lo stile reale (0.01 = poco, 0.1 = tanto). 0.05 è un buon mix.

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*L)).astype(int)
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # Scambia lo spettro di ampiezza (stile) mantenendo la fase (struttura)
    fft_src = torch.fft.fftn(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.fftn(trg_img.clone(), dim=(-2, -1))

    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg, _ = torch.abs(fft_trg), torch.angle(fft_trg)

    # Muta la parte a bassa frequenza dell'ampiezza sorgente con quella target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # Ricostruisci l'immagine
    fft_src_ = amp_src_ * torch.exp(1j * pha_src)
    src_in_trg = torch.fft.ifftn(fft_src_, dim=(-2, -1))
    src_in_trg = torch.real(src_in_trg)

    return src_in_trg

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Trasformazioni base
    trans = transforms.Compose([
        transforms.Resize((512, 1024)), # Resize standard
        transforms.ToTensor(),
    ])

    src_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(('.jpg', '.png'))])
    
    # Se la cartella target non esiste o è vuota, crea un avviso
    if not os.path.exists(TRG_DIR) or not os.listdir(TRG_DIR):
        print(f"ERRORE: Devi mettere delle immagini reali di riferimento in {TRG_DIR}")
        print("Basta anche copiare 5-6 immagini da un dataset reale (Cityscapes/GSV) lì dentro.")
        return

    trg_files = sorted([f for f in os.listdir(TRG_DIR) if f.endswith(('.jpg', '.png'))])
    
    print(f"Inizio FDA (Beta={BETA})...")

    for i, src_name in enumerate(src_files):
        # Carica Source (GTA)
        src_img = Image.open(os.path.join(SRC_DIR, src_name)).convert('RGB')
        src_tensor = trans(src_img).unsqueeze(0) # Add batch dim

        # Carica Target (Reale) - Ne pesca una a caso o ciclica
        trg_name = trg_files[i % len(trg_files)]
        trg_img = Image.open(os.path.join(TRG_DIR, trg_name)).convert('RGB')
        trg_tensor = trans(trg_img).unsqueeze(0)

        # Applica FDA
        with torch.no_grad():
            out_tensor = FDA_source_to_target(src_tensor, trg_tensor, L=BETA)

        # Salva
        save_path = os.path.join(OUTPUT_DIR, src_name)
        save_image(out_tensor, save_path)
        print(f"[{i+1}/{len(src_files)}] Convertita {src_name} usando stile di {trg_name}")

    print("Finito.")

if __name__ == "__main__":
    main()