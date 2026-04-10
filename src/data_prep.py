import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def setup_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def prepare_dataloaders(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                        batch_size=16, 
                        max_length=128):
    print("Caricamento del dataset tweet_eval...")
    dataset = load_dataset("tweet_eval", "sentiment")

    print(f"[INFO] Download del tokenizzatore per {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )


    print("[INFO] Tokenizzazione del dataset in corso...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # DataLoaders
    print("Creazione dei PyTorch DataLoaders...")
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

    print("Pre-processing completato")
    return train_dataloader, eval_dataloader, test_dataloader, tokenizer

device = setup_device()
print(f"Esecuzione sul device: {device}")
    
train_dl, eval_dl, test_dl, tokenizer = prepare_dataloaders()
    
for batch in train_dl:
   print("\n[VERIFICA DEL BATCH PER IL MODELLO]")
   print(f"Dimensioni input_ids (Batch Size, Max Length): {batch['input_ids'].shape}")
   print(f"Dimensioni attention_mask: {batch['attention_mask'].shape}") 
   print(f"Dimensioni labels: {batch['labels'].shape}")
   break
