import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os

# --- Model Setup ---

# Locally Hosted Model (Flan-T5 is a great choice for CPU, and it's instruction-tuned)
local_translator = pipeline("text2text-generation", model="google/flan-t5-base")

# API-based Model (Zephyr 7B - High Quality, runs on remote server)
# Use the environment variable from your Hugging Face Space secret
api_token = os.environ.get("HF")
api_client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=api_token)

# --- Single Prediction Function ---
def get_translation(simple_text, model_choice, temperature):
    # Input validation
    if not simple_text or not simple_text.strip():
        return "⚠️ Please enter a sentence."

    # A simple but effective prompt for both models.
    # The more powerful API model will interpret it better.
    prompt = (
        f"Rephrase the following sentence into elegant, formal, and slightly archaic English, "
        f"as if spoken by a monarch. Do not use modern slang or emojis.\n\n"
        f"Sentence: {simple_text}"
    )

    if model_choice == "Locally Hosted":
        try:
            # For a Seq2Seq model like Flan-T5, do_sample and temperature are standard generation parameters.
            result = local_translator(
                prompt,
                max_new_tokens=100,
                do_sample=True, # enable sampling for creativity
                temperature=temperature # pass the slider value directly
            )[0]["generated_text"]
            return result.strip()
        except Exception as e:
            return f"❌ Local model error: {e}"

    elif model_choice == "API-Based":
        # Check if the token is available for the API client
        if not api_token:
            return "❌ API token not found. Please set the 'HF' secret in your Space settings."
        try:
            # Use the API client to generate text
            response = api_client.text_generation(
                prompt=prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=temperature,
                return_full_text=False
            )
            # The InferenceClient returns a string directly if return_full_text is False
            return response.strip()
        except Exception as e:
            return f"❌ API Error: {e}"

    else:
        return "⚠️ Please select a model version."

# --- Gradio UI ---
with gr.Blocks(title="Elegant English") as demo:
    gr.Markdown("# 👑 Elegant English Translator")
    gr.Markdown("Convert casual sentences into elegant, royal-style English.")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            ["Locally Hosted", "API-Based"],
            label="Select Model",
            value="Locally Hosted"
        )
    with gr.Row():
        input_textbox = gr.Textbox(
            lines=2,
            placeholder="Enter a simple sentence...",
            label="Simple English"
        )
        output_textbox = gr.Textbox(label="Elegant English Translation")
    with gr.Row():
        temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Creativity / Temperature"
        )
    with gr.Row():
        submit_btn = gr.Button("Translate", variant="primary")

    submit_btn.click(
        fn=get_translation,
        inputs=[input_textbox, model_dropdown, temperature_slider],
        outputs=output_textbox
    )

if __name__ == "__main__":
    demo.launch()