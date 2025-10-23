import os
from pathlib import Path

def visit_images(root_dir, extensions={'.jpg', '.jpeg', '.png'}):
    """
    Generatore che visita tutte le immagini nel dataset e restituisce percorsi relativi
    rispetto alla root assoluta.
    """
    root_path = Path(root_dir).resolve()
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            ext = Path(file).suffix.lower()
            if ext in extensions:
                full_path = Path(dirpath) / file
                yield full_path.relative_to(root_path)

def count_images(root_dir, extensions={'.jpg', '.jpeg', '.png'}):
    """
    Conta tutte le immagini nel dataset usando visit_images.
    """
    return sum(1 for _ in visit_images(root_dir, extensions))

dataset_root = '/home/lcantagallo/VPR-GTAV2Real/dataset/GTAV'
    
# Conta immagini
total = count_images(dataset_root)
print(f"Totale immagini: {total}") # Output: Totale immagini: 20919