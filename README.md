---
title: Sentiment Analysis MLOps
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
python_version: "3.10"
pinned: false
---


# Progetto di Sentiment Analysis e Architettura MLOps - MachineInnovators Inc.

Questo repository ospita il codice sorgente e l'infrastruttura architetturale per un sistema avanzato di Sentiment Analysis. Il progetto nasce con il duplice obiettivo di sviluppare un modello di Natural Language Processing (NLP) altamente accurato per l'analisi dei testi provenienti dai social media e, contestualmente, di ingegnerizzarne il ciclo di vita seguendo i paradigmi fondanti del Machine Learning Operations (MLOps). L'intera infrastruttura è stata progettata per garantire scalabilità, riproducibilità e automazione dei processi di test e rilascio.

## 1. Scelta Tecnologica: Da FastText all'Architettura Transformer (RoBERTa)

La specifica iniziale del progetto prevedeva l'implementazione di un modello basato su **FastText**. Tuttavia, a seguito di un'analisi sui requisiti di accuratezza necessari per elaborare il linguaggio dei social media, ho ritenuto opportuno effettuare un aggiornamento architetturale verso un modello basato su Transformer, nello specifico **RoBERTa** (versione `cardiffnlp/twitter-roberta-base-sentiment-latest`).

A differenza dei modelli tradizionali che analizzano le parole singolarmente o per n-grammi, l'architettura RoBERTa sfrutta il meccanismo di *Self-Attention* bidirezionale. Questo permette al modello di comprendere il contesto semantico globale della frase, risultando significativamente più efficace nell'individuare sarcasmo, ironia e sfumature di significato complesse tipiche dei tweet.

A livello infrastrutturale, per rispettare le best practice di versionamento del codice e i limiti di spazio imposti da Git (100 MB), i pesi del modello (che ammontano a circa 475 MB) sono stati esclusi dal repository tramite `.gitignore`. Il sistema è configurato per scaricare i pesi dinamicamente da un Model Registry esterno (**Hugging Face Hub**) al momento dell'esecuzione.

## 2. Ingegnerizzazione del Codice e Sviluppo Modulare

Per favorire la manutenibilità e la transizione da un ambiente di pura ricerca (Jupyter Notebook) a uno di produzione, il codice sorgente è stato fattorizzato in script Python modulari, contenuti all'interno della directory `src/`:

* **`eda.py` (Exploratory Data Analysis):** Automatizza l'analisi preliminare del dataset, valutando la distribuzione delle classi di sentiment (Positivo, Neutro, Negativo) e le caratteristiche testuali. Questo passaggio è cruciale per prevenire bias di addestramento dovuti a classi sbilanciate.
* **`data_prep.py`:** Gestisce la pipeline di pre-processing. Si occupa della pulizia del testo e, soprattutto, dell'istanziazione del tokenizer specifico di RoBERTa, trasformando le stringhe di testo in tensori PyTorch ottimizzati per l'addestramento.
* **`train.py`:** Contiene la logica di addestramento e valutazione. Implementa un sistema di monitoraggio statico che, al termine del processo, calcola e registra metriche fondamentali come l'Accuracy globale e il F1-Score (valutato in modalità macro per tenere conto di eventuali sbilanciamenti minori tra le classi).

## 3. Continuous Integration e Automazione (GitHub Actions)

Il cuore della metodologia MLOps di questo progetto risiede nell'implementazione di una pipeline di Continuous Integration, definita nel file `.github/workflows/mlops_pipeline.yml`.

L'obiettivo di questa pipeline è fungere da *Quality Gate* automatizzato. Ad ogni push sul branch principale, l'infrastruttura di GitHub alloca un ambiente virtuale sterile in cui:

* Viene configurato un ambiente Python controllato (versione 3.10).
* Vengono installate le dipendenze deterministiche definite nel file `requirements.txt`.
* Viene eseguita l'intera suite di script (`eda.py`, `data_prep.py`, `train.py`) come *Smoke Test*.

Questo approccio garantisce che nessuna modifica al codice possa essere integrata se genera regressioni o errori di compilazione, proteggendo l'integrità del sistema in produzione.

## 4. Deploy Disaccoppiato e Interfaccia Utente

In conformità con i principi di microservizi e separazione delle responsabilità (decoupling), il repository GitHub è stato dedicato esclusivamente al versionamento del codice e ai test. Il deployment dell'applicazione finale è stato invece affidato a un'infrastruttura cloud esterna ottimizzata per il Machine Learning: **Hugging Face Spaces**.

L'applicazione in produzione utilizza la libreria **Gradio** per fornire un'interfaccia utente web accessibile e interattiva. Il server istanzia dinamicamente la pipeline di inferenza, recupera il modello RoBERTa e permette l'elaborazione del sentiment in tempo reale, restituendo all'utente una valutazione probabilistica (Confidence Score) associata alla classe di appartenenza.

# Progetto di Sentiment Analysis e Architettura MLOps - MachineInnovators Inc.

> Modello su Hugging Face: [Sentiment Analysis App su Hugging Face Spaces](https://grinko-sentiment-analysis-mlops.hf.space)


