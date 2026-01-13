import yaml
import subprocess
import time
import sys

# I semi per le 3 run
SEEDS = [42, 100, 1234]
CONFIG_FILE = "pipeline.yaml"

def update_seed(seed):
    """Legge lo yaml, cambia il seed e lo risalva mantenendo la struttura"""
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    config["seed"] = seed
    
    # Assicuriamoci che save_dir sia corretto
    config["save_dir"] = "cocktail"
    
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"[BATCH] Config aggiornata con Seed: {seed}")

def main():
    print(f"=== AVVIO BATCH EXPERIMENT (3 RUNS) ===")
    
    for i, seed in enumerate(SEEDS):
        run_name = f"Run {i+1}/3 (Seed {seed})"
        print(f"\n{'-'*60}")
        print(f"STARTING {run_name}")
        print(f"{'-'*60}\n")
        
        # 1. Aggiorna il seed nel file yaml
        update_seed(seed)
        
        # 2. Lancia la pipeline completa (Train -> Test -> Recall -> Telegram)
        # Usa sys.executable per essere sicuri di usare lo stesso python env
        result = subprocess.run([sys.executable, "pipeline.py"])
        
        if result.returncode != 0:
            print(f"!!! ERRORE IN {run_name} !!! Interrompo il batch.")
            break
            
        print(f"COMPLETATO {run_name}")
        time.sleep(5) # Piccola pausa per sicurezza filesystem

    print("\n=== BATCH COMPLETO ===")

if __name__ == "__main__":
    main()