import gradio as gr
from transformers import pipeline

# ✅ Use FLAN-T5 for instruction-following tasks
local_translator = pipeline("text2text-generation", model="google/flan-t5-base")

def get_translation(simple_text, model_choice):
    if not simple_text.strip():
        return "⚠️ Please enter a sentence."

    if model_choice == "Locally Hosted":
        prompt = (
            f"Rephrase this sentence into elegant, formal, and slightly archaic English:\n\n{simple_text}"
        )
        result = local_translator(prompt, max_new_tokens=128)
        return result[0]["generated_text"]

    elif model_choice == "API-Based":
        return "❌ API currently not available — set HF token first."

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
