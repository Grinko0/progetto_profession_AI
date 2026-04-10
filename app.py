import gradio as gr
from transformers import pipeline

analyzer = None
LOG_FILE = "flagged/logs.csv"

def predict(text):
    global analyzer
    if analyzer is None:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        analyzer = pipeline("sentiment-analysis", model=model_name)
        
    if not text.strip(): return "Inserisci del testo."
    results = analyzer(text)[0]
    return f"Sentiment: {results['label']} (Confidenza: {results['score']:.2%})"

# Funzione per generare il Grafico di Monitoraggio
 def get_monitoring_chart():
    if not os.path.exists(LOG_FILE):
        return px.annotation_mode_template = "Nessun dato di monitoraggio ancora disponibile."
    df = pd.read_csv(LOG_FILE)
    if 'flag' in df.columns:
        fig = px.bar(df, x='flag', title="Monitoraggio Feedback Utenti", 
                     labels={'flag': 'Tipo di Feedback', 'count': 'Numero di segnalazioni'},
                     color='flag')
        return fig
    else:
        return None

with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ Sentiment AI Radar & Monitoring")
    
    with gr.Tab("Analisi Sentiment"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Inserisci un tweet in inglese")
                submit_btn = gr.Button("Analizza", variant="primary")
            with gr.Column():
                output_text = gr.Text(label="Risultato")
        
        # Bottoni di monitoraggio
        submit_btn.click(predict, input_text, output_text)
        
        # Sistema di flagging integrato che salva nel CSV
        callback = gr.CSVLogger()
        callback.setup([input_text, output_text], "flagged")
        
        gr.Markdown("### Valuta la qualità del modello (Monitoraggio Attivo)")
        with gr.Row():
            flag_err = gr.Button("Errato")
            flag_ok = gr.Button("Corretto")
        
        # Quando clicchi, salva nel log.csv
        flag_err.click(lambda *args: callback.flag(args, flag_option="Errore"), [input_text, output_text], None)
        flag_ok.click(lambda *args: callback.flag(args, flag_option="OK"), [input_text, output_text], None)

    with gr.Tab("Dashboard Monitoraggio"):
        gr.Markdown("### Statistiche di Performance in Tempo Reale")
        plot = gr.Plot()
        refresh_btn = gr.Button("Aggiorna Grafici")
        
        # Aggiorna il grafico leggendo il CSV
        refresh_btn.click(get_monitoring_chart, None, plot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
