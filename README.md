# MLOps Sentiment Analysis - MachineInnovators Inc.

Progetto di analisi del sentiment scalabile basato su **RoBERTa** e metodologie **MLOps**.

## Architettura del Modello
- **Base Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Motivazione:** Utilizzo dell'architettura Transformer per catturare il contesto semantico complesso nei tweet rispetto ai tradizionali modelli FastText.

## Pipeline MLOps
- **CI/CD:** Implementata tramite GitHub Actions.
- **Workflow automatizzato:** 1. Installazione ambiente (Python 3.10)
  2. EDA automatizzata
  3. Pre-processing e Tokenizzazione
  4. Training / Smoke Test
- **Monitoraggio:** Il sistema verifica le metriche (Accuracy, F1-Score) ad ogni push sul branch main.

## Come riprodurre in locale
1. `pip install -r requirements.txt`
2. `python src/eda.py`
3. `python src/train.py`
