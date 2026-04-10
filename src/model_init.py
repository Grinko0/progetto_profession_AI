import torch
from transformers import AutoModelForSequenceClassification

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def initialize_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", num_labels=3):
    device = setup_device()
    print(f"Inizializzazione del modello {model_name}...")
    
    # Caricamento del modello 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    )
    
    # Spostamento del modello sulla GPU 
    model = model.to(device)
    print(f"[INFO] Modello caricato con successo sul device: {device}")
    
    # calcolo del numero di parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[INFO] Parametri totali del modello: {total_params:,}")
    print(f"[INFO] Parametri addestrabili (che verranno modificati): {trainable_params:,}")
    
    return model, device


model, device = initialize_model()
    
print("\n[VERIFICA ARCHITETTURALE]")
print("Struttura dell'ultimo layer (Classification Head) che farà le previsioni:")
print(model.classifier)
