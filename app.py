import gradio as gr
from transformers import pipeline
import pandas as pd
import plotly.express as px
import os

# 1. Caricamento Modello
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

# Funzione per pulire le caselle
def reset_inputs():
    return "", ""

# Funzione per il Grafico
def get_monitoring_chart():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        if 'flag' in df.columns:
            counts = df['flag'].value_counts().reset_index()
            counts.columns = ['Feedback', 'Conteggio']
            fig = px.bar(counts, x='Feedback', y='Conteggio', 
                         title="Monitoraggio Performance",
                         color='Feedback',
                         color_discrete_map={'Errore': '#ef553b', 'OK': '#00cc96'})
            return fig
    except:
        return None
    return None

# 3. Interfaccia
with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ Sentiment AI Radar & Monitoring")
    
    with gr.Tab("Analisi Sentiment"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Inserisci un tweet in inglese", placeholder="Scrivi qui...")
                with gr.Row():
                    submit_btn = gr.Button("Analizza", variant="primary")
                    reset_btn = gr.Button("Reset")
            with gr.Column():
                output_text = gr.Text(label="Risultato")
                feedback_msg = gr.Markdown("") 
        
        submit_btn.click(predict, input_text, output_text)
        reset_btn.click(reset_inputs, None, [input_text, output_text])
        
        # Sistema di monitoraggio
        callback = gr.CSVLogger()
        callback.setup([input_text, output_text], "flagged")
        
        gr.Markdown("### Valuta la qualità del modello")
        with gr.Row():
            flag_err = gr.Button("❌ Errato")
            flag_ok = gr.Button("✅ Corretto")
        
        # Salvataggio con messaggio di feedback
        flag_err.click(lambda *args: (callback.flag(args, flag_option="Errore"), "✅ Feedback 'Errato' registrato!"), 
                      [input_text, output_text], [feedback_msg, feedback_msg])
        
        flag_ok.click(lambda *args: (callback.flag(args, flag_option="OK"), "✅ Feedback 'Corretto' registrato!"), 
                     [input_text, output_text], [feedback_msg, feedback_msg])

    with gr.Tab("Dashboard Monitoraggio"):
        gr.Markdown("### Statistiche di Performance")
        refresh_btn = gr.Button("🔄 Aggiorna Grafico")
        plot = gr.Plot()
        
        refresh_btn.click(get_monitoring_chart, None, plot)

if __name__ == "__main__":
    demo.launch(
