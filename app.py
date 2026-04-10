import gradio as gr
from transformers import pipeline

analyzer = None

def predict(text):
    global analyzer
    if analyzer is None:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        analyzer = pipeline("sentiment-analysis", model=model_name)
        
    if not text.strip(): return "Inserisci del testo."
    results = analyzer(text)[0]
    return f"Sentiment: {results['label']} (Confidenza: {results['score']:.2%})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Inserisci un tweet in inglese"),
    outputs=gr.Text(label="Risultato Analisi"),
    allow_flagging="manual", 
    flagging_options=["Predizione Errata", "Sentiment Non Chiaro", "Tutto OK"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
