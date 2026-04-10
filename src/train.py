import torch
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer
import evaluate
from tqdm.auto import tqdm
from data_prep import prepare_dataloaders
from model_init import initialize_model

def train_model(epochs=3, max_batches_per_epoch=10):
    # Modello e Dati
    print("[INFO] Preparazione Dati e Modello in corso...")
    train_dl, eval_dl, _, _ = prepare_dataloaders(batch_size=16)
    model, device = initialize_model()

    # Ottimizzatore
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Scheduler 
    num_training_steps = epochs * (max_batches_per_epoch if max_batches_per_epoch else len(train_dl))
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    # Inizializzazione Metriche (Accuracy e F1-Macro)
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    print(f"\n[INFO] INIZIO ADDESTRAMENTO ({epochs} Epoche)")
    
    # TRAINING LOOP 
    for epoch in range(epochs):
        print(f"\n=== Epoca {epoch + 1}/{epochs} ===")
        model.train() 
        
        train_loss = 0.0
        progress_bar = tqdm(range(max_batches_per_epoch if max_batches_per_epoch else len(train_dl)), desc="Training")
        
        for i, batch in enumerate(train_dl):
            if max_batches_per_epoch and i >= max_batches_per_epoch:
                break 
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward Pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward Pass
            loss.backward()
            
            # aggiorniamo i pesi
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad() 
            
            train_loss += loss.item()
            progress_bar.update(1)
            
        print(f"Loss media di Training: {train_loss / (i + 1):.4f}")

        # Validazione
        model.eval() 
        eval_progress_bar = tqdm(range(max_batches_per_epoch if max_batches_per_epoch else len(eval_dl)), desc="Evaluation")
        
        for i, batch in enumerate(eval_dl):
            if max_batches_per_epoch and i >= max_batches_per_epoch:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch)
            
            # Estraiamo la previsione con la probabilità più alta 
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # metriche
            metric_acc.add_batch(predictions=predictions, references=batch["labels"])
            metric_f1.add_batch(predictions=predictions, references=batch["labels"])
            
            eval_progress_bar.update(1)

        acc_result = metric_acc.compute()
        f1_result = metric_f1.compute(average="macro")
        
        print(f"[RISULTATI EPOCA {epoch + 1}]")
        print(f"Accuracy: {acc_result['accuracy']:.4f}")
        print(f"F1-Score (Macro): {f1_result['f1']:.4f}")

    # 5. Salvataggio del modello addestrato
    print("\nSalvataggio del modello e del tokenizer in './saved_model'...")
    model.save_pretrained("./saved_model")
   
    AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").save_pretrained("./saved_model")
    print("Addestramento completato!")

if __name__ == "__main__":
    # Avviamo il training loop. Di default è limitato a 10 batch/epoca per test su CPU.
    train_model(epochs=1, max_batches_per_epoch=10)
