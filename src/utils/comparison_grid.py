import os
import matplotlib.pyplot as plt
from PIL import Image

def create_three_column_grid():
    # --- PERCORSI ---
    gta_dir = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GTAVSamples"
    # La cartella dove hai messo le immagini reali di riferimento (GSV/Mapillary)
    ref_dir = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/GSVSamples"
    # La cartella dell'output FDA
    fda_dir = "/home/lcantagallo/VPR-GTAV2Real/src/dataset/Output_Real_FDA"
    report_dir = "/home/lcantagallo/VPR-GTAV2Real/src/exp_003/note/reportImages"

    os.makedirs(report_dir, exist_ok=True)

    # Liste file ordinate
    gta_files = sorted([f for f in os.listdir(gta_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    ref_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Prendiamo solo le prime 3 per il collage
    num_images = 3
    gta_files_top = gta_files[:num_images]
    
    if not gta_files_top or not ref_files:
        print("ERRORE: Mancano immagini in input (GTA) o in reference (RealSamples).")
        return

    # Setup Griglia 3x3 (più larga per far stare 3 colonne)
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 5 * num_images))
    
    # Gestione caso limite se num_images fosse 1
    if num_images == 1: axes = [axes]

    print(f"Generazione griglia 3x3 per {len(gta_files_top)} righe...")

    for i, filename in enumerate(gta_files_top):
        # 1. Path Immagine GTA (Geometria)
        gta_path = os.path.join(gta_dir, filename)
        
        # 2. Path Immagine Reference (Stile)
        # NOTA: Usiamo la stessa logica ciclica dello script di generazione per trovare il match
        ref_idx = i % len(ref_files)
        ref_filename = ref_files[ref_idx]
        ref_path = os.path.join(ref_dir, ref_filename)

        # 3. Path Immagine Output (Risultato)
        fda_path = os.path.join(fda_dir, filename)

        # Verifica esistenza di tutti e tre i file
        if os.path.exists(fda_path) and os.path.exists(ref_path):
            img_gta = Image.open(gta_path).convert('RGB')
            img_ref = Image.open(ref_path).convert('RGB')
            img_fda = Image.open(fda_path).convert('RGB')

            # Selettori assi
            ax_gta = axes[i][0] if num_images > 1 else axes[0]
            ax_ref = axes[i][1] if num_images > 1 else axes[1]
            ax_fda = axes[i][2] if num_images > 1 else axes[2]

            # Colonna 1: GTA
            ax_gta.imshow(img_gta)
            ax_gta.set_title("1. Content Source (GTA)", fontsize=11, fontweight='bold')
            ax_gta.axis("off")

            # Colonna 2: Reference
            # Resize per visualizzazione coerente se le ref hanno size diverse
            ax_ref.imshow(img_ref.resize(img_gta.size)) 
            ax_ref.set_title(f"2. Style Reference ({ref_filename[:10]}...)", fontsize=11)
            ax_ref.axis("off")

            # Colonna 3: Result
            ax_fda.imshow(img_fda)
            ax_fda.set_title("3. Result (FDA Mix)", fontsize=11, fontweight='bold')
            ax_fda.axis("off")
            
            # Aggiunge un "+" e un "=" visivi tra i grafici (opzionale, per effetto "somma")
            if i == 0: # Solo sulla prima riga per pulizia
                fig.text(0.36, 0.90, "+", fontsize=30, ha='center', va='center')
                fig.text(0.64, 0.90, "=", fontsize=30, ha='center', va='center')

        else:
            print(f"SKIP: Mancano file per la riga {filename} (controlla output FDA o cartella reference)")

    plt.tight_layout()
    # Aggiusta il layout per far spazio ai simboli + e = se usati
    plt.subplots_adjust(top=0.92) 
    
    # --- MODIFICA NOME FILE ---
    save_path = os.path.join(report_dir, "GTA_plus_Real_eq_FDA.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Collage 3-colonne salvato in: {save_path}")

if __name__ == "__main__":
    create_three_column_grid()