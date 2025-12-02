import requests
import yaml

def load_config(path="telegram.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def send_telegram_message(message):
    config = load_config("telegram.yaml")
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data) 

if __name__ == "__main__":
    send_telegram_message("Prova di notifica Telegram dal modulo notify.py")
