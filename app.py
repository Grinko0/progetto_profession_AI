import gradio as gr
from transformers import pipeline
import pandas as pd
import plotly.express as px
import os

# Caricamento Modello (Lazy Loading)
analyzer = None
LOG_FILE = "flagged/log.csv" 

def predict(text):
    global analyzer
    if analyzer is None:
        analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    if not text.strip(): 
        return "Inserisci del testo."
    results = analyzer(text)[0]
    return f"Sentiment: {results['label']} (Confidenza: {results['score']:.2%})"

# Funzione per generare il Grafico (Assicurati che l'indentazione sia corretta qui!)
def get_monitoring_chart():
    if not os.path.exists(LOG_FILE):
        return None
    
    try:
        df = pd.read_csv(LOG_FILE)
        if 'flag' in df.columns:
            # Conta le occorrenze di ogni tipo di feedback
            counts = df['flag'].value_counts().reset_index()
            counts.columns = ['Feedback', 'Conteggio']
            fig = px.bar(counts, x='Feedback', y='Conteggio', 
                         title="Monitoraggio Performance (Feedback Utenti)",
                         color='Feedback',
                         color_discrete_map={'Errore': '#ef553b', 'OK': '#00cc96'})
            return fig
    except Exception as e:
        print(f"Errore nella lettura dei log: {e}")
    return None

# Interfaccia con TABS
with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ Sentiment AI Radar & Monitoring")
    
    with gr.Tab("Analisi Sentiment"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Inserisci un tweet in inglese", placeholder="Scrivi qui...")
                submit_btn = gr.Button("Analizza", variant="primary")
            with gr.Column():
                output_text = gr.Text(label="Risultato")
        
        submit_btn.click(predict, input_text, output_text)
        
        # Sistema di monitoraggio (Flagging)
        callback = gr.CSVLogger()
        callback.setup([input_text, output_text], "flagged")
        
        gr.Markdown("### Valuta la qualità del modello")
        with gr.Row():
            flag_err = gr.Button("❌ Errato")
            flag_ok = gr.Button("✅ Corretto")
        
        # Salvataggio nel file CSV
        flag_err.click(lambda *args: callback.flag(args, flag_option="Errore"), [input_text, output_text], None)
        flag_ok.click(lambda *args: callback.flag(args, flag_option="OK"), [input_text, output_text], None)

    with gr.Tab("Dashboard Monitoraggio"):
        gr.Markdown("### Statistiche di Performance")
        refresh_btn = gr.Button("Aggiorna Grafico")
        plot = gr.Plot()
        
        # Aggiorna il grafico al click
        refresh_btn.click(get_monitoring_chart, None, plot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
