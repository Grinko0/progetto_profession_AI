"""
Modulo: eda.py
Obiettivo: Eseguire l'Analisi Esplorativa dei Dati (EDA) sul dataset di addestramento.

SCELTE PROGETTUALI E ARCHITETTURALI:
1. Dataset Scelto: 'tweet_eval' (configurazione 'sentiment').
   - Perché: È il benchmark standard accademico e industriale per la sentiment analysis su Twitter. 
     Contiene tweet reali, già pre-processati per rimuovere artefatti inutili, ed è perfettamente 
     allineato con il modello RoBERTa scelto (cardiffnlp/twitter-roberta-base-sentiment-latest) 
     che è stato originariamente addestrato proprio su questi dati.
2. Utilizzo di Hugging Face 'datasets':
   - Perché: Rispetto al download manuale di file CSV, la libreria 'datasets' usa un backend 
     basato su Apache Arrow. Questo permette di caricare dataset enormi mappandoli in memoria (memory-mapping)
     senza saturare la RAM del Codespace, garantendo scalabilità se in futuro passeremo a dataset più grandi.
3. Astrazione Hardware (CUDA/CPU):
   - Perché: Anche se l'EDA si appoggia a CPU (tramite Pandas), inizializzare il controllo hardware 
     fin da ora garantisce che l'infrastruttura PyTorch sia pronta per le Fasi di training.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def setup_device():
    """
    Rileva e configura l'acceleratore hardware disponibile.
    
    Teoria: Il Deep Learning richiede calcolo matriciale massivo. PyTorch utilizza 'cuda' 
    per inviare questi calcoli alle GPU NVIDIA. Su GitHub Codespaces standard avremo 'cpu', 
    ma questa funzione garantisce che il codice non si rompa se eseguiamo il deploy su 
    macchine provviste di GPU.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Hardware Device selezionato: {device}")
    return device

def perform_eda(dataset_name="tweet_eval", config="sentiment", output_dir="plots"):
    """
    Esegue l'estrazione delle statistiche vitali del dataset.
    
    Scelte Analitiche:
    - Analisi del Bilanciamento delle Classi: Se un dataset ha il 90% di tweet positivi, 
      il modello imparerà a predire sempre "positivo" ignorando le feature del testo (Mode Collapse). 
      Questa analisi ci dirà se avremo bisogno di tecniche di bilanciamento (es. pesi nella Loss function) nella Fase 4.
    - Distribuzione della Lunghezza: I modelli Transformer (come RoBERTa) hanno un limite rigido 
      di token in input (es. 512). Analizzare la lunghezza media ci permette di troncare o paddare 
      i testi in modo efficiente nella Fase 2, risparmiando memoria computazionale.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Caricamento del dataset {dataset_name} ({config})...")
    dataset = load_dataset(dataset_name, config)
    df_train = dataset['train'].to_pandas()
    
    # 1. Controllo Integrità dei Dati
    missing_values = df_train.isnull().sum()
    print("\n[INFO] Valori mancanti per colonna (Dovrebbero essere 0):")
    print(missing_values)

    # 2. Analisi Bilanciamento Classi
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_train, x='label', palette='mako')
    plt.title('Bilanciamento delle Classi nel Dataset di Training')
    plt.xlabel('Sentiment (0: Negativo, 1: Neutro, 2: Positivo)')
    plt.ylabel('Conteggio dei Tweet')
    plt.savefig(os.path.join(output_dir, 'class_balance.png'))
    plt.close()
    print(f"[INFO] Grafico bilanciamento classi salvato in {output_dir}/class_balance.png")

    # 3. Analisi Lunghezza Sequenze (in parole)
    df_train['word_length'] = df_train['text'].apply(lambda test: len(str(test).split()))
    
    plt.figure(figsize=(8, 6))
    sns.histplot(df_train['word_length'], bins=40, kde=True, color='teal')
    plt.title('Distribuzione della Lunghezza dei Tweet')
    plt.xlabel('Numero di Parole per Tweet')
    plt.ylabel('Frequenza')
    plt.savefig(os.path.join(output_dir, 'sequence_length.png'))
    plt.close()
    print(f"[INFO] Grafico lunghezza sequenze salvato in {output_dir}/sequence_length.png")

    print("\n[INFO] Statistiche descrittive sulla lunghezza (utile per il token padding):")
    print(df_train['word_length'].describe())

if __name__ == "__main__":
    device = setup_device()
    perform_eda()