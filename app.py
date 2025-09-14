import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os

# --- Model and Client Setup ---

# ✅ Locally hosted model (better than GPT-2 for instruction following)
# If you want something very small & fast, use distilgpt2
# If you have GPU, you can switch to a small instruct-tuned model for better results
local_translator = pipeline("text-generation", model="distilgpt2")

# ✅ API client setup (with token)
api_token = os.environ.get("HF")
api_client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)

# --- Single Prediction Function ---
def get_translation(simple_text, model_choice):
    if not simple_text.strip():
        return "⚠️ Please enter a sentence."

    if model_choice == "Locally Hosted":
        # Few-shot prompt to guide GPT-like model
        prompt = (
            "Rephrase the following sentence into an elegant, formal, and slightly archaic English style, "
            "as if spoken by a monarch. Do not use modern slang or emojis.\n\n"
            "Simple: Hello Everyone! Lets go for a ride\n"
            "Elegant: Greetings, all. I propose we embark upon a most agreeable journey.\n\n"
            f"Simple: {simple_text}\nElegant:"
        )

        generated = local_translator(
            prompt,
            max_new_tokens=60,
            pad_token_id=50256
        )[0]["generated_text"]

        # ✅ Strip the original prompt and return only the continuation
        output = generated[len(prompt):].strip()
        return output if output else "(No meaningful output from local model)"

    elif model_choice == "API-Based":
        prompt = (
            f"Rephrase the sentence into elegant, formal, and slightly archaic English. "
            f"Sentence: '{simple_text}'"
        )
        try:
            response = api_client.text_generation(
                prompt=prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                return_full_text=False
            )
            # ✅ Handle response object
            if isinstance(response, list):
                return response[0].generated_text.strip()
            elif hasattr(response, "generated_text"):
                return response.generated_text.strip()
            else:
                return str(response)
        except Exception as e:
            return f"❌ API Error: {e}"

    else:
        return "⚠️ Please select a model version."

# --- Gradio Interface ---
with gr.Blocks(title="Elegant English") as demo:
    gr.Markdown("# 👑 Elegant English Translator")
    gr.Markdown("Turn everyday speech into elegant, royal English.")

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
        submit_btn = gr.Button("Translate", variant="primary")

    submit_btn.click(
        fn=get_translation,
        inputs=[input_textbox, model_dropdown],
        outputs=output_textbox
    )

if __name__ == "__main__":
    demo.launch()
