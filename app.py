import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os

# --- Model and Client Setup ---

# ✅ Locally Hosted Model (SmolLM2-135M-Instruct)
# A small, instruction-tuned model for on-device/local execution
try:
    local_translator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Failed to load local model: {e}")
    local_translator = None

# ✅ API-based Model (Llama-3.1-8B-Instruct)
# A powerful, instruction-tuned model for the API-based solution.
# Ensure you have accepted the license on the model's Hugging Face page.
api_token = os.environ.get("HF")
api_client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct", token=api_token)

# --- Single Prediction Function ---
def get_translation(simple_text, model_choice, temperature):
    """
    Translates a simple sentence into elegant English using either a local or API-based model.
    """
    if not simple_text or not simple_text.strip():
        return "⚠️ Please enter a sentence."

    # General prompt for instruction-tuned models
    prompt = (
        f"Rephrase the following sentence into an elegant, formal, and slightly archaic English style, "
        f"as if spoken by a monarch. Do not use modern slang or emojis.\n\n"
        f"Sentence: {simple_text}"
    )

    if model_choice == "Locally Hosted":
        if not local_translator:
            return "❌ Local model failed to load. Please check logs."
        try:
            result = local_translator(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=temperature
            )[0]["generated_text"]
            
            # The model returns the prompt plus the completion, so we strip the prompt.
            output = result.split("Sentence:")[-1].split("Answer:")[-1].strip()
            return output
        except Exception as e:
            return f"❌ Local model error: {e}"

    elif model_choice == "API-Based":
        if not api_token:
            return "❌ API token not found. Please set the 'HF' secret in your Space settings."
        try:
            response = api_client.text_generation(
                prompt=prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=temperature,
                return_full_text=False
            )
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