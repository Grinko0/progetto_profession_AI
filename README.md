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

# Progetto Sentiment Analysis MLOps - MachineInnovators Inc.

> **Live Demo:** [Applicazione su Hugging Face Spaces](https://grinko-sentiment-analysis-mlops.hf.space)

Benvenuti nel repository del mio progetto. L'obiettivo di questo lavoro è fornire a MLOps Innovators Inc. un sistema automatizzato per l'analisi del sentiment sui social media, permettendo all'azienda di rispondere rapidamente ai feedback degli utenti e gestire in modo proattivo la propria reputazione online. 

Invece di fermarmi al semplice addestramento di un modello, ho ingegnerizzato l'intero ciclo di vita del software seguendo le metodologie MLOps. Di seguito spiego nel dettaglio le scelte implementative affrontate nelle varie fasi del progetto.

---

## Fase 1: Implementazione del Modello e Gestione Dati

**La scelta del modello (FastText vs RoBERTa)**
Le linee guida iniziali menzionavano l'uso di un modello basato su **FastText**. Tuttavia, il riferimento specifico fornito nei requisiti puntava al modello `cardiffnlp/twitter-roberta-base-sentiment-latest`. Ho deciso di seguire questa seconda indicazione implementando **RoBERTa**, un modello basato su architettura Transformer. 
Questa scelta si è rivelata molto più solida per i social media: a differenza di FastText, RoBERTa utilizza la *Self-Attention* bidirezionale, riuscendo a comprendere il contesto dell'intera frase. Questo è essenziale per cogliere sfumature, ironia e sarcasmo, mantenendo alta l'accuratezza predittiva.

**Dataset e Preparazione**
Ho utilizzato un dataset pubblico contenente testi estratti dai social media, già etichettati in tre classi (Positivo, Neutro, Negativo). Il codice per la preparazione dei dati (`src/data_prep.py`) si occupa di pulire i testi e tokenizzarli nel formato tensoriale richiesto dal Transformer.

---

## Fase 2: Creazione della Pipeline CI/CD

Per garantire l'affidabilità del codice, ho sviluppato una pipeline automatizzata utilizzando **GitHub Actions**. Ad ogni push sul repository, la pipeline configura l'ambiente, installa le dipendenze e lancia i test di integrazione per verificare che l'addestramento e il deploy possano avvenire senza errori.

**L'errore affrontato e risolto (Lo "Smoke Test")**
Durante lo sviluppo, è emerso un problema: l'addestramento del modello nella pipeline CI era limitato a soli 10 batch. Questo andava benissimo come *smoke test* per verificare che il codice non andasse in crash, ma non permetteva un addestramento reale del modello.
Per risolvere questo problema in modo elegante, ho reso lo script di training (`train.py`) **"context-aware"**. Ho inserito un controllo sulla variabile d'ambiente `CI=true` (iniettata automaticamente da GitHub). 
* Se il codice gira su GitHub, esegue lo smoke test veloce (10 batch) per risparmiare risorse e tempo.
* Se il codice gira in locale o in produzione, il limite viene rimosso automaticamente e parte il *full training* sull'intero dataset.

---

## Fase 3: Deploy e Monitoraggio Continuo della Reputazione

**Deploy su Hugging Face**
Ho disaccoppiato il versionamento dal deploy: il codice vive su GitHub, ma l'applicazione è ospitata su **Hugging Face Spaces**. Tramite la libreria **Gradio**, ho creato un'interfaccia web che permette di testare il modello in tempo reale.

**Sistema di Monitoraggio e HitL (Human-in-the-Loop)**
Per soddisfare il requisito del monitoraggio continuo della reputazione, non bastava fare inferenza. Ho aggiunto una dashboard interattiva all'app, permettendo agli utenti di fornire un feedback sulle predizioni (cliccando su "Corretto" o "Errato"). Questo mi permette di valutare continuamente le performance del modello sul campo e di accorgermi subito se la percezione dell'azienda cambia (Data Drift).

**L'errore affrontato e risolto (I log fantasma)**
Su Hugging Face, i server cloud si resettano spesso (ephemeral storage), il che mi faceva perdere i file CSV dei log di monitoraggio. Inoltre, i grafici Plotly si bloccavano spesso a causa delle policy sugli iframe.
Per risolvere, ho scritto una funzione personalizzata in `pandas` che forza la scrittura su disco del file CSV ad ogni singolo feedback. Per il lato visivo, ho sostituito Plotly con il componente nativo `gr.BarPlot` di Gradio, affiancandogli una tabella dati (`gr.Dataframe`) per poter ispezionare visivamente i log crudi in tempo reale, assicurandomi che il salvataggio funzioni sempre.

---

## Verso il Retraining Automatico

Il sistema di monitoraggio che ho implementato pone le basi esatte per il requisito del **Retraining del Modello**. Tutti i testi che vengono etichettati come "Errati" tramite i bottoni di feedback vengono salvati nel log. Questi dati non sono scarti, ma costituiscono il nuovo dataset etichettato che verrà utilizzato per ri-addestrare dinamicamente il modello. In questo modo, l'algoritmo potrà adattarsi continuamente ai nuovi slang e ai cambiamenti nei comportamenti degli utenti sui social media, mantenendo l'accuratezza sempre ai massimi livelli.

---

## Consegna e Materiale
* Il codice sorgente documentato è interamente disponibile in questo repository GitHub.
* La consegna formale del progetto viene effettuata tramite il notebook **Google Colab** allegato, che contiene il link diretto a questo repository e le istruzioni per l'avvio.

