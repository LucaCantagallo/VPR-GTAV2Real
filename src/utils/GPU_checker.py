import torch

if torch.cuda.is_available():
    print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
    print(f"Numero di GPU: {torch.cuda.device_count()}")
else:
    print("Nessuna GPU disponibile, userà la CPU")
