import os
import subprocess
import sys
from settings import get_test_work_dir
from utils import load_params
from notify import send_telegram_message


def run_step(name, command):
    print(f"\n[STEP] Avvio {name}...\n{'-'*50}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Errore durante {name}. Interrompo la pipeline.")
        send_telegram_message(f"Errore durante {name}. Interrompo la pipeline.")
        sys.exit(result.returncode)
    print(f"{name} completato con successo!\n{'='*50}")

def read_file(path):
    with open(path, "r") as f:
        return f.read()

if __name__ == "__main__":
    send_telegram_message("Avvio della pipeline di training.")

    config_file = "./pipeline.yaml"
    params = load_params(config_file)

    yaml_content = read_file("./pipeline.yaml")
    send_telegram_message(f"Contenuto di pipeline.yaml:\n{yaml_content}")
    run_step("TRAINING", "python train.py")

    run_step("TESTING", "python test.py")
    run_step("RECALL", "python recall.py")
    print("\nPipeline completata con successo!")

    test_work_dir = get_test_work_dir(params, experiments_dir="./experiments")
    recall_file = os.path.join(test_work_dir, "recall_cm_best_loss.txt")
    content = read_file(recall_file)
    send_telegram_message(f"File di recall:\n{content}")


    send_telegram_message("Pipeline completata con successo!")
    
