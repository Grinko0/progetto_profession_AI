import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def setup_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Hardware Device selezionato: {device}")
    return device

def perform_eda(dataset_name="tweet_eval", config="sentiment", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Caricamento del dataset {dataset_name} ({config})...")
    dataset = load_dataset(dataset_name, config)
    df_train = dataset['train'].to_pandas()
    
    # Controllo Integrità dei Dati
    missing_values = df_train.isnull().sum()
    print("\n[INFO] Valori mancanti per colonna (Dovrebbero essere 0):")
    print(missing_values)

    # Analisi Bilanciamento Classi
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_train, x='label', palette='mako')
    plt.title('Bilanciamento delle Classi nel Dataset di Training')
    plt.xlabel('Sentiment (0: Negativo, 1: Neutro, 2: Positivo)')
    plt.ylabel('Conteggio dei Tweet')
    plt.savefig(os.path.join(output_dir, 'class_balance.png'))
    plt.close()
    print(f"Grafico bilanciamento classi salvato in {output_dir}/class_balance.png")

    # Analisi Lunghezza Sequenze 
    df_train['word_length'] = df_train['text'].apply(lambda test: len(str(test).split()))
    
    plt.figure(figsize=(8, 6))
    sns.histplot(df_train['word_length'], bins=40, kde=True, color='teal')
    plt.title('Distribuzione della Lunghezza dei Tweet')
    plt.xlabel('Numero di Parole per Tweet')
    plt.ylabel('Frequenza')
    plt.savefig(os.path.join(output_dir, 'sequence_length.png'))
    plt.close()
    print(f"[INFO] Grafico lunghezza sequenze salvato in {output_dir}/sequence_length.png")

    print("\nStatistiche descrittive sulla lunghezza (utile per il token padding):")
    print(df_train['word_length'].describe())

device = setup_device()
perform_eda()
