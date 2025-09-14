import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os

# --- Model Setup ---

# ✅ Local Instruction-Tuned Model (CPU-friendly, small but good)
local_translator = pipeline("text2text-generation", model="google/flan-t5-base")

# ✅ API-based Model (Zephyr 7B - High Quality)
api_token = os.environ.get("HF")  # set this before running: export HF=your_hf_token
api_client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=api_token)

# --- Single Prediction Function ---
def get_translation(simple_text, model_choice, temperature):
    if not simple_text.strip():
        return "⚠️ Please enter a sentence."

    prompt = (
        f"Rephrase the following sentence into elegant, formal, and slightly archaic English, "
        f"as if spoken by a monarch. Do not use modern slang or emojis.\n\n"
        f"Sentence: {simple_text}"
    )

    if model_choice == "Locally Hosted":
        try:
            # Flan-T5 does not use temperature directly, but we can vary outputs with num_beams
            result = local_translator(
                prompt,
                max_new_tokens=100,
                num_beams=1 if temperature > 0.6 else 4,  # more beams = more deterministic
                do_sample=True if temperature > 0.3 else False,
                top_p=1.0,
                temperature=temperature
            )[0]["generated_text"]
            return result.strip()
        except Exception as e:
            return f"❌ Local model error: {e}"

    elif model_choice == "API-Based":
        if not api_token:
            return "❌ No API token set. Please run:\n\nexport HF=your_hf_token"
        try:
            response = api_client.text_generation(
                prompt=prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=temperature,
                return_full_text=False
            )
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
