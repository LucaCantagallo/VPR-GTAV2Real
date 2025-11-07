import os
import sys

def rename_experiment_folders(path: str):
    # Controlla se il path è una directory valida
    if not os.path.isdir(path):
        print(f"❌ Il path '{path}' non è una cartella valida.")
        return

    # Controlla che il genitore si chiami 'experiments'
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    if parent != "experiments":
        print(f"❌ Il path '{path}' non si trova dentro una cartella 'experiments'. Nessuna modifica effettuata.")
        return

    # Ottieni tutte le sottocartelle
    subfolders = [
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ]
    
    if not subfolders:
        print("ℹ️ Nessuna sottocartella trovata.")
        return

    # Ordina numericamente (solo se tutti i nomi sono numerici)
    try:
        subfolders_sorted = sorted(subfolders, key=lambda x: int(x))
    except ValueError:
        print("⚠️ Alcune cartelle non hanno nomi numerici. Non verranno rinominate.")
        return

    print(f"🔍 Cartelle trovate in '{path}': {subfolders_sorted}")

    # Rinomina in ordine
    for new_index, folder_name in enumerate(subfolders_sorted):
        old_path = os.path.join(path, folder_name)
        new_path = os.path.join(path, str(new_index))

        if old_path != new_path:
            print(f"🔄 Rinomino '{folder_name}' -> '{new_index}'")
            os.rename(old_path, new_path)

    print("✅ Rinomina completata con successo!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rename_experiments.py <path_alla_cartella_da_rinominare>")
    else:
        rename_experiment_folders(sys.argv[1])
