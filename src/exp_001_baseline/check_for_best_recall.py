import os
import re
from glob import glob

# Directory di partenza (puoi modificare se vuoi)
ROOT_DIR = "/home/lcantagallo/VPR-GTAV2Real/src/exp_001_baseline"

# Regex per estrarre i valori delle recall
RECALL_REGEX = re.compile(
    r"Top-(\d+)\s+recall\s+\(day/night/avg\):\s+[\d.]+\s*/\s*[\d.]+\s*/\s*([\d.]+)"
)

def parse_final_evaluation(file_path):
    recalls = {}
    with open(file_path, "r") as f:
        text = f.read()
        for match in RECALL_REGEX.finditer(text):
            k = int(match.group(1))
            avg_recall = float(match.group(2))
            recalls[k] = avg_recall
    return recalls

def compare_recalls(a, b):
    """Confronta due dizionari di recall con priorità Top-50 > 10 > 5 > 1"""
    for key in [50, 10, 5, 1]:
        va, vb = a.get(key, 0), b.get(key, 0)
        if va != vb:
            return va - vb
    return 0

def find_best_model(root_dir):
    # Trova tutti i final_evaluation.txt nelle sottocartelle
    files = glob(os.path.join(root_dir, "experiments", "*", "*", "final_evaluation.txt"))
    if not files:
        print("❌ Nessun file final_evaluation.txt trovato.")
        return

    best_path = None
    best_recalls = None

    for file_path in files:
        recalls = parse_final_evaluation(file_path)
        if not recalls:
            continue
        if best_recalls is None or compare_recalls(recalls, best_recalls) > 0:
            best_path = file_path
            best_recalls = recalls

    if best_path:
        print("Miglior modello trovato:")
        print(f"{best_path}")
        print(f"Top-1: {best_recalls.get(1,0):.4f}, Top-5: {best_recalls.get(5,0):.4f}, Top-10: {best_recalls.get(10,0):.4f}, Top-50: {best_recalls.get(50,0):.4f}")
    else:
        print("Nessun file valido trovato.")

if __name__ == "__main__":
    find_best_model(ROOT_DIR)
