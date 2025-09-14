import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os # <-- Import the os library

# --- Model and Client Setup ---
# Locally hosted model setup
local_translator = pipeline("text2text-generation", model="t5-small")

# API-based client setup (using a token from a secret)
api_token = os.environ.get("HF_TOKEN") # <-- Get the token from the environment
if not api_token:
    # Handle the case where the token is not set (e.g., local testing)
    api_token = "your_placeholder_token_for_local_testing" 
api_client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)

# --- The Single Prediction Function ---
def get_translation(simple_text, model_choice):
    if model_choice == "Locally Hosted":
        # ... (your existing local model logic) ...
        prompt = f"Translate '{simple_text}' into elegant English."
        regal_text = local_translator(prompt, max_length=100, num_beams=4)[0]['generated_text']
        return regal_text

    elif model_choice == "API-Based":
        # Logic for the API-based model
        prompt = (
            f"Rephrase the sentence into elegant, formal, and archaic English. Sentence: '{simple_text}'"
        )
        response = api_client.text_generation(prompt=prompt, max_new_tokens=150)
        return response.strip()

    else:
        return "Please select a model version."

# --- Gradio Interface with gr.Blocks() ---
with gr.Blocks(title="Regal Rhetoric") as demo:
    gr.Markdown("# Regal Rhetoric: Choose Your Translator")
    gr.Markdown("Select a model to translate simple English into an elegant, royal style.")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            ["Locally Hosted", "API-Based"],
            label="Select Model",
            value="Locally Hosted"
        )

    with gr.Row():
        input_textbox = gr.Textbox(lines=2, placeholder="Enter a simple sentence...", label="Simple English")
        output_textbox = gr.Textbox(label="Regal English Translation")

    with gr.Row():
        submit_btn = gr.Button("Translate")

    submit_btn.click(
        fn=get_translation,
        inputs=[input_textbox, model_dropdown],
        outputs=output_textbox
    )

if __name__ == "__main__":
    demo.launch()