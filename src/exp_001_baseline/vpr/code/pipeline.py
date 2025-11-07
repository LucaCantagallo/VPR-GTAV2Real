import subprocess
import sys

def run_step(name, command):
    print(f"\n[STEP] Avvio {name}...\n{'-'*50}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Errore durante {name}. Interrompo la pipeline.")
        sys.exit(result.returncode)
    print(f"{name} completato con successo!\n{'='*50}")

if __name__ == "__main__":
    # Puoi modificare o commentare i passaggi che vuoi saltare
    run_step("TRAINING", "python train.py")
    run_step("TESTING", "python test.py")
    run_step("RECALL", "python recall.py")

    print("\nPipeline completata con successo!")
