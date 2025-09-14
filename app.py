import gradio as gr
from transformers import pipeline

# Load the model directly from the Hugging Face Hub
translator = pipeline("text2text-generation", model="t5-small")

def translate_to_regal_english(simple_text):
    """
    Translates a simple English sentence into a more regal style.
    """
    prompt = f"Translate the following simple sentence into elegant, formal, and slightly archaic English: '{simple_text}'"
    regal_text = translator(prompt, max_length=100, num_beams=4)[0]['generated_text']
    return regal_text

# Create the Gradio interface
iface = gr.Interface(
    fn=translate_to_regal_english,
    inputs=gr.Textbox(lines=2, placeholder="Enter a simple English sentence...", label="Simple English"),
    outputs=gr.Textbox(label="Regal English Translation"),
    title="Enhanced and Fine Tuned English",
    description="A sophisticated tool to translate everyday phrases into elegant, royal rhetoric."
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()