"""
Modulo: model_init.py
Obiettivo: Inizializzazione del modello RoBERTa e configurazione del Classification Head.

SCELTE PROGETTUALI E ARCHITETTURALI:
1. AutoModelForSequenceClassification: Questa classe di Hugging Face è fondamentale. 
   Invece di scaricare solo il "corpo" del modello (che estrae le feature linguistiche), 
   questa classe aggiunge automaticamente una "testa" (un Linear Layer finale) configurata 
   per restituire esattamente 'num_labels' (nel nostro caso 3).
2. Gestione Device (CUDA): Un modello Transformer pesa centinaia di Megabyte. Spostare i suoi 
   pesi (matrici) sulla memoria VRAM della GPU tramite 'model.to(device)' è il passaggio più 
   critico per l'accelerazione hardware. Senza questo, il training richiederebbe giorni anziché minuti.
3. id2label / label2id: Aggiungiamo questa mappatura per comodità MLOps. In fase di inferenza, 
   il modello non restituirà uno sterile "0", ma direttamente l'etichetta "NEGATIVE", 
   facilitando la creazione dell'API e dell'interfaccia utente.
"""

import torch
from transformers import AutoModelForSequenceClassification

def setup_device():
    """Rileva l'hardware per garantire compatibilità cross-platform."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def initialize_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", num_labels=3):
    """
    Scarica i pesi pre-addestrati, istanzia l'architettura e la sposta sul device corretto.
    """
    device = setup_device()
    print(f"[INFO] Inizializzazione del modello {model_name}...")
    
    # Caricamento del modello con la testa di classificazione
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    )
    
    # Spostamento del modello sulla GPU (o CPU se in fallback)
    model = model.to(device)
    print(f"[INFO] Modello caricato con successo sul device: {device}")
    
    # Analisi computazionale: calcolo del numero di parametri
    # Questo ci dà un'idea di quanto sia "pesante" il modello da addestrare in RAM/VRAM
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[INFO] Parametri totali del modello: {total_params:,}")
    print(f"[INFO] Parametri addestrabili (che verranno modificati): {trainable_params:,}")
    
    return model, device

if __name__ == "__main__":
    # Eseguiamo l'inizializzazione per verificarne il funzionamento
    model, device = initialize_model()
    
    print("\n[VERIFICA ARCHITETTURALE]")
    print("Struttura dell'ultimo layer (Classification Head) che farà le previsioni:")
    # Stampiamo solo il classificatore per confermare che l'output sia impostato a 3
    print(model.classifier)