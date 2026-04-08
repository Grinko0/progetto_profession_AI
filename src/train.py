"""
Modulo: train.py
Obiettivo: Implementazione del Training Loop e validazione per il modello RoBERTa.

SCELTE PROGETTUALI:
1. PyTorch Training Loop nativo: Non usiamo l'astrazione 'Trainer' di Hugging Face. 
   Scrivere il loop a mano ci dà il 100% di controllo per l'MLOps (es. inserire log custom, 
   gestire W&B, o limitare i batch per testare la CI/CD su macchine senza GPU).
2. AdamW + Linear Scheduler: Ottimizza l'aggiornamento dei pesi. Lo scheduler 
   riduce linearmente il learning rate verso lo zero man mano che ci avviciniamo alla fine, 
   permettendo al modello di "assestarsi" sul minimo globale della funzione di costo.
"""

import torch
from torch.optim import AdamW
from transformers import get_scheduler
import evaluate
from tqdm.auto import tqdm

# Importiamo i moduli scritti nelle fasi precedenti
from data_prep import prepare_dataloaders
from model_init import initialize_model

def train_model(epochs=3, max_batches_per_epoch=10):
    """
    Esegue l'addestramento e la validazione del modello.
    NOTA MLOps: 'max_batches_per_epoch' è impostato a 10 per permettere 
    il test su CPU in tempi rapidi. Per un training reale, impostare a None.
    """
    # 1. Setup Modello e Dati
    print("[INFO] Preparazione Dati e Modello in corso...")
    train_dl, eval_dl, _, _ = prepare_dataloaders(batch_size=16)
    model, device = initialize_model()

    # 2. Setup Ottimizzatore
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 3. Setup Scheduler (Pianificatore del Learning Rate)
    # Calcoliamo quanti step totali faremo
    num_training_steps = epochs * (max_batches_per_epoch if max_batches_per_epoch else len(train_dl))
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    # 4. Inizializzazione Metriche (Accuracy e F1-Macro)
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    print(f"\n[INFO] INIZIO ADDESTRAMENTO ({epochs} Epoche)")
    
    # === TRAINING LOOP ===
    for epoch in range(epochs):
        print(f"\n=== Epoca {epoch + 1}/{epochs} ===")
        model.train() # Mettiamo il modello in modalità addestramento (attiva Dropout, ecc.)
        
        train_loss = 0.0
        progress_bar = tqdm(range(max_batches_per_epoch if max_batches_per_epoch else len(train_dl)), desc="Training")
        
        for i, batch in enumerate(train_dl):
            if max_batches_per_epoch and i >= max_batches_per_epoch:
                break # Interrompiamo presto per testare la pipeline su CPU
                
            # Spostiamo i tensori del batch sulla CPU/GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward Pass: passiamo i dati al modello
            outputs = model(**batch)
            loss = outputs.loss # La CrossEntropyLoss è calcolata in automatico!
            
            # Backward Pass: calcolo dei gradienti (derivate)
            loss.backward()
            
            # Step di ottimizzazione: aggiorniamo i pesi
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad() # Resettiamo i gradienti per il batch successivo
            
            train_loss += loss.item()
            progress_bar.update(1)
            
        print(f"Loss media di Training: {train_loss / (i + 1):.4f}")

        # === EVALUATION LOOP (Validazione) ===
        model.eval() # Mettiamo il modello in modalità valutazione (disattiva Dropout)
        eval_progress_bar = tqdm(range(max_batches_per_epoch if max_batches_per_epoch else len(eval_dl)), desc="Evaluation")
        
        for i, batch in enumerate(eval_dl):
            if max_batches_per_epoch and i >= max_batches_per_epoch:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Disattiviamo il calcolo dei gradienti per risparmiare memoria ed energia
            with torch.no_grad():
                outputs = model(**batch)
            
            # Estraiamo la previsione con la probabilità più alta (argmax)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Aggiungiamo i risultati al calcolatore di metriche
            metric_acc.add_batch(predictions=predictions, references=batch["labels"])
            metric_f1.add_batch(predictions=predictions, references=batch["labels"])
            
            eval_progress_bar.update(1)

        # Calcolo metriche finali per l'epoca
        acc_result = metric_acc.compute()
        f1_result = metric_f1.compute(average="macro")
        
        print(f"[RISULTATI EPOCA {epoch + 1}]")
        print(f"Accuracy: {acc_result['accuracy']:.4f}")
        print(f"F1-Score (Macro): {f1_result['f1']:.4f}")

    # 5. Salvataggio del modello addestrato
    print("\n[INFO] Salvataggio del modello e del tokenizer in './saved_model'...")
    model.save_pretrained("./saved_model")
    # Il tokenizer lo carichiamo di nuovo al volo per salvarlo assieme al modello
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").save_pretrained("./saved_model")
    print("[INFO] Addestramento completato!")

if __name__ == "__main__":
    # Avviamo il training loop. Di default è limitato a 10 batch/epoca per test su CPU.
    train_model(epochs=1, max_batches_per_epoch=10)