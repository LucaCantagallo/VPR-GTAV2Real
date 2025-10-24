import os
from pathlib import Path

from PIL import Image


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

def print_image_sizes(root_dir, extensions={'.jpg', '.jpeg', '.png'}, max_items=None):
    """
    Stampa le dimensioni (width x height) di ogni immagine nel dataset.
    Puoi limitare il numero di immagini stampate con max_items.
    """
    root_path = Path(root_dir).resolve()
    count = 0

    for rel_path in visit_images(root_dir, extensions):
        abs_path = root_path / rel_path
        try:
            with Image.open(abs_path) as img:
                width, height = img.size
                print(f"{rel_path}: {width}×{height}")
        except Exception as e:
            print(f"{rel_path}: errore nell'apertura ({e})")

        count += 1
        if max_items and count >= max_items:
            break


dataset_root = '/home/lcantagallo/VPR-GTAV2Real/dataset/GTAV'

# Conta immagini
# print(f"Totale immagini: {count_images(dataset_root)}") # Output: Totale immagini: 20919


print_image_sizes(dataset_root, max_items=100)  # Solo le prime 10 immagini
