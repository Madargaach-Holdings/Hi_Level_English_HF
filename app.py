import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient
import os

# --- Model and Client Setup ---

# ✅ Locally Hosted Model (nouamanetazi/cover-letter-t5-base)
# A small, fine-tuned model for local execution.
try:
    local_generator = pipeline("text2text-generation", model="nouamanetazi/cover-letter-t5-base")
except Exception as e:
    print(f"Failed to load local model: {e}")
    local_generator = None

# ✅ API-based Model (Llama-3.1-8B-Instruct)
# A powerful, general-purpose model accessed via the Inference API.
api_token = os.environ.get("HF")
api_client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct", token=api_token)

# --- Single Prediction Function ---
def generate_cover_letter(resume_text, job_description, model_choice):
    """
    Generates a cover letter based on resume and job description.
    """
    if not resume_text or not job_description:
        return "⚠️ Please provide both a resume and a job description."

    # A general prompt for instruction-tuned models
    prompt = (
        f"You are a cover letter writing assistant. Write a professional cover letter "
        f"that highlights how the applicant's resume is a perfect fit for the job description. "
        f"Be concise and professional. The applicant's resume is below:\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"The job description is below:\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Cover Letter:"
    )

    if model_choice == "Locally Hosted":
        if not local_generator:
            return "❌ Local model failed to load. Check Space logs."
        try:
            # The local model uses a specific prompt format
            input_text = f"resume: {resume_text} job_description: {job_description}"
            cover_letter = local_generator(input_text, max_new_tokens=500)[0]['generated_text']
            return cover_letter.strip()
        except Exception as e:
            return f"❌ Local model error: {e}"

    elif model_choice == "API-Based":
        if not api_token:
            return "❌ API token not found. Please set the 'HF' secret in your Space settings."
        try:
            # Pass the detailed prompt to the API client
            response = api_client.text_generation(
                prompt=prompt,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                return_full_text=False
            )
            return response.strip()
        except Exception as e:
            return f"❌ API Error: {e}"

    else:
        return "⚠️ Please select a model version."

# --- Gradio UI ---
with gr.Blocks(title="Cover Letter Generator") as demo:
    gr.Markdown("# 🚀 AI-Powered Cover Letter Generator")
    gr.Markdown("Select a model to generate a custom cover letter from your resume and a job description.")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            ["Locally Hosted", "API-Based"],
            label="Select Model",
            value="Locally Hosted"
        )
    with gr.Row():
        resume_input = gr.Textbox(lines=5, label="Your Resume (Text)")
        job_description_input = gr.Textbox(lines=5, label="Job Description (Text)")
    
    submit_btn = gr.Button("Generate Cover Letter", variant="primary")
    output_textbox = gr.Textbox(label="Generated Cover Letter", lines=10)

    submit_btn.click(
        fn=generate_cover_letter,
        inputs=[resume_input, job_description_input, model_dropdown],
        outputs=output_textbox
    )

if __name__ == "__main__":
    demo.launch()