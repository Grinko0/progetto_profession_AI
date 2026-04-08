"""
Modulo: data_prep.py
Obiettivo: Tokenizzazione e preparazione dei PyTorch DataLoader per RoBERTa.

SCELTE PROGETTUALI E ARCHITETTURALI:
1. Tokenizzatore Pre-addestrato: Utilizziamo AutoTokenizer per caricare il vocabolario
   esatto di 'cardiffnlp/twitter-roberta-base-sentiment-latest'.
2. Truncation e Padding (max_length=128): Scelta derivata dall'EDA. Ottimizza
   l'uso della memoria (VRAM/RAM) evitando matrici sparse inutilmente grandi.
3. Formattazione Tensoriale: Convertiamo i dati in tensori PyTorch ('torch').
   Nota su CUDA: I DataLoader generano tensori sulla CPU (nella RAM di sistema). 
   Lo spostamento sulla GPU ('cuda') avverrà riga per riga durante il training loop 
   nella Fase 4, per evitare di saturare la memoria video della GPU prima del tempo.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def setup_device():
    """Rileva l'hardware per garantire compatibilità cross-platform."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def prepare_dataloaders(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                        batch_size=16, 
                        max_length=128):
    """
    Scarica il dataset, lo tokenizza e restituisce i PyTorch DataLoader.
    """
    # 1. Caricamento del dataset testuale
    print("[INFO] Caricamento del dataset tweet_eval...")
    dataset = load_dataset("tweet_eval", "sentiment")

    # 2. Inizializzazione del Tokenizzatore
    print(f"[INFO] Download del tokenizzatore per {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Funzione di Tokenizzazione
    def tokenize_function(examples):
        """
        Tokenizza un batch di testi. 
        - padding='max_length': Aggiunge zeri (token ID di pad) per arrivare a max_length.
        - truncation=True: Taglia il testo in eccesso se supera max_length.
        """
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # 4. Mappatura sul dataset (batched=True rende l'operazione multi-threading e velocissima)
    print("[INFO] Tokenizzazione del dataset in corso...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 5. Formattazione per l'addestramento con PyTorch
    # Rimuoviamo la colonna testuale originale, il modello legge solo 'input_ids' e 'attention_mask'
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # Rinominiamo 'label' in 'labels' (convenzione rigorosa richiesta da Hugging Face per il calcolo della Loss)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Convertiamo le liste Python in tensori PyTorch
    tokenized_datasets.set_format("torch")

    # 6. Creazione dei DataLoaders
    print("[INFO] Creazione dei PyTorch DataLoaders...")
    # Shuffle solo sul train set per la discesa del gradiente stocastica (SGD)
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

    print("[INFO] Pre-processing completato con successo!")
    return train_dataloader, eval_dataloader, test_dataloader, tokenizer

if __name__ == "__main__":
    device = setup_device()
    print(f"[INFO] Esecuzione sul device: {device}")
    
    # Eseguiamo la pipeline per testarla
    train_dl, eval_dl, test_dl, tokenizer = prepare_dataloaders()
    
    # Estraiamo un singolo batch per verificarne la forma matematica (Shape)
    for batch in train_dl:
        print("\n[VERIFICA DEL BATCH PER IL MODELLO]")
        # input_ids: i numeri che rappresentano i token
        print(f"Dimensioni input_ids (Batch Size, Max Length): {batch['input_ids'].shape}")
        # attention_mask: 1 per i token reali, 0 per i token di padding
        print(f"Dimensioni attention_mask: {batch['attention_mask'].shape}")
        # labels: le etichette di sentiment
        print(f"Dimensioni labels: {batch['labels'].shape}")
        break # Interrompiamo il ciclo dopo aver stampato il primo batch