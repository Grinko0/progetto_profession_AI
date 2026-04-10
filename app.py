import gradio as gr
from transformers import pipeline
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Configurazione Percorsi
LOG_DIR = "flagged"
LOG_FILE = os.path.join(LOG_DIR, "logs.csv")

# Crea la cartella se non esiste all'avvio
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Caricamento Modello
analyzer = None

def predict(text):
    global analyzer
    if analyzer is None:
        analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    if not text.strip(): 
        return "Inserisci del testo."
    results = analyzer(text)[0]
    return f"Sentiment: {results['label']} (Confidenza: {results['score']:.2%})"

# FUNZIONE DI LOGGING 
def save_feedback(input_text, output_text, feedback_type):
    if not input_text or not output_text or "Sentiment:" not in output_text:
        return "Errore: Analizza un tweet prima di inviare il feedback!"
    
    try:
        new_data = {
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "input": [input_text],
            "output": [output_text],
            "flag": [str(feedback_type)] 
        }
        df_new = pd.DataFrame(new_data)
        
        if not os.path.isfile(LOG_FILE):
            df_new.to_csv(LOG_FILE, index=False)
        else:
            df_new.to_csv(LOG_FILE, mode='a', header=False, index=False)
        
        return f"Feedback '{feedback_type}' registrato correttamente!"
    except Exception as e:
        return f"Errore nel salvataggio: {e}"

def get_monitoring_chart():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        if not df.empty and 'flag' in df.columns:
            counts = df['flag'].value_counts().reset_index()
            counts.columns = ['Feedback', 'Conteggio']
            fig = px.bar(counts, x='Feedback', y='Conteggio', 
                         title="Monitoraggio Performance (Feedback Utenti)",
                         color='Feedback',
                         color_discrete_map={'Errore': '#ef553b', 'OK': '#00cc96'})
            return fig
    except Exception as e:
        print(f"Errore grafico: {e}")
    return None

def reset_inputs():
    return "", "", "Pronto per l'analisi."

# Interfaccia
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
                feedback_msg = gr.Markdown("Pronto per l'analisi.") 
        
        submit_btn.click(predict, input_text, output_text)
        
        gr.Markdown("### Valuta la qualità del modello (Monitoraggio Attivo)")
        with gr.Row():
            flag_err = gr.Button("Errato")
            flag_ok = gr.Button("Corretto")
        
        # Salvataggio feedback
        flag_err.click(fn=lambda i, o: save_feedback(i, o, "Errore"), 
                      inputs=[input_text, output_text], 
                      outputs=feedback_msg)
        
        flag_ok.click(fn=lambda i, o: save_feedback(i, o, "OK"), 
                     inputs=[input_text, output_text], 
                     outputs=feedback_msg)
        
        reset_btn.click(reset_inputs, None, [input_text, output_text, feedback_msg])

    with gr.Tab("Dashboard Monitoraggio"):
        gr.Markdown("### Statistiche di Performance in Tempo Reale")
        refresh_btn = gr.Button("🔄 Aggiorna Grafico", variant="primary")
        plot = gr.Plot()
        refresh_btn.click(get_monitoring_chart, None, plot)

if __name__ == "__main__":
    demo.launch()
