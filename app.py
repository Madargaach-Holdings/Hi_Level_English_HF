import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os

# --- Model and Client Setup ---
# Locally hosted model setup
# Using gpt2, a small, open-source model suitable for local execution
local_translator = pipeline("text-generation", model="openai-community/gpt2")

# API-based client setup (using a token from a secret)
api_token = os.environ.get("HF")
api_client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)

# --- The Single Prediction Function ---
def get_translation(simple_text, model_choice):
    if model_choice == "Locally Hosted":
        # Logic for the locally hosted model
        # Using a few-shot prompt to guide the text generation
        prompt = (
            f"Rephrase the following sentence into an elegant, formal, and slightly archaic English style, as if spoken by a monarch. Do not use modern slang or emojis.\n\n"
            f"Simple: Hello Everyone! Lets go for a ride\n"
            f"Elegant: Greetings, all. I propose we embark upon a most agreeable journey.\n\n"
            f"Simple: {simple_text}\n"
            f"Elegant:"
        )
        
        # Generate the translated text. The 'text-generation' pipeline will continue the prompt.
        generated_text = local_translator(
            prompt, 
            max_length=150, 
            num_return_sequences=1, 
            pad_token_id=50256  # This is needed for GPT2
        )[0]['generated_text']

        # Clean up the output to only return the "Elegant" part
        # This is necessary because text-generation pipelines return the full prompt + completion
        output = generated_text.split("Elegant:")[-1].strip()
        return output

    elif model_choice == "API-Based":
        # Logic for the API-based model
        prompt = (
            f"Rephrase the sentence into elegant, formal, and archaic English. Sentence: '{simple_text}'"
        )
        try:
            response = api_client.text_generation(
                prompt=prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                return_full_text=False
            )
            return response.strip()
        except Exception as e:
            return f"An error occurred with the API call: {e}"

    else:
        return "Please select a model version."

# --- Gradio Interface with gr.Blocks() ---
with gr.Blocks(title="Elegant English") as demo:
    gr.Markdown("# Elegant English: Choose Your Translator")
    gr.Markdown("Select a model to translate simple English into an elegant, royal style.")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            ["Locally Hosted", "API-Based"],
            label="Select Model",
            value="Locally Hosted"
        )

    with gr.Row():
        input_textbox = gr.Textbox(lines=2, placeholder="Enter a simple sentence...", label="Simple English")
        output_textbox = gr.Textbox(label="Elegant English Translation")

    with gr.Row():
        submit_btn = gr.Button("Translate")

    submit_btn.click(
        fn=get_translation,
        inputs=[input_textbox, model_dropdown],
        outputs=output_textbox
    )

if __name__ == "__main__":
    demo.launch()