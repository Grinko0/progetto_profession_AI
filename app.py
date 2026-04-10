import gradio as gr
from transformers import pipeline
import datetime
import csv
import os

LOG_FILE = "monitoring_logs.csv"

def log_inference(text, sentiment, confidence):
    """Registra ogni predizione per il monitoraggio continuo"""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Text", "Sentiment", "Confidence"])
        writer.writerow([datetime.datetime.now(), text, sentiment, confidence])

# MODELLO 
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
analyzer = pipeline("sentiment-analysis", model=model_name)

def predict(text):
    if not text.strip(): return "Inserisci del testo."
    
    results = analyzer(text)[0]
    label = results['label']
    score = results['score']
    

    log_inference(text, label, score)
    
    return f"Sentiment: {label} (Confidenza: {score:.2%})"

# INTERFACCIA GRADIO 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Inserisci un tweet in inglese", placeholder="Esempio: I love MLOps!"),
    outputs=gr.Text(label="Risultato Analisi"),
    title="MachineInnovators - Sentiment AI Radar",
    description="Sistema di monitoraggio live delle performance del modello.",
    allow_flagging="manual", # Permette agli utenti di segnalare errori (Monitoring)
    flagging_options=["Errore Sentiment", "Bassa Confidenza"]
)

demo.launch()
