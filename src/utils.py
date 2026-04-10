import torch

def setup_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo selezionato: {device}")
    return device
