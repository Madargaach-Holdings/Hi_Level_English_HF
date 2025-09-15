import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import InferenceClient
import json
import time
import re
from datetime import datetime
import hashlib


# Initialize the Hugging Face Inference Client for API-based approach
HF_KEY = os.getenv("HF_API_TOKEN")  # get from environment
if not HF_KEY:
    raise ValueError("❌ No Hugging Face API key found. Please set HF_API_TOKEN env var.")

client = InferenceClient(
    "google/vaultgemma-1b",
    token=HF_KEY
)

class SecureChatAnalytics:
    def __init__(self):
        self.session_id = self.generate_session_id()
        self.analysis_history = []
        
    def generate_session_id(self):
        """Generate a unique session ID for privacy tracking"""
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    def sanitize_input(self, text):
        """Basic input sanitization for security"""
        # Remove potential harmful patterns
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # Keep only safe characters
        return text.strip()
    
    def analyze_sentiment_tone(self, text):
        """Analyze sentiment and tone using API-based model"""
        try:
            sanitized_text = self.sanitize_input(text)
            
            prompt = f"""Analyze the following text for sentiment, tone, and security concerns:

Text: "{sanitized_text}"

Provide analysis in this format:
- Sentiment: [Positive/Negative/Neutral]
- Tone: [Professional/Casual/Aggressive/Friendly/etc.]
- Security Risk: [Low/Medium/High]
- Key Themes: [list main themes]
- Privacy Score: [1-10, where 10 is most private]

Analysis:"""
            
            start_time = time.time()
            
            # Use the API-based approach
            response = client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                return_full_text=False
            )
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Store analysis in history
            analysis_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": self.session_id,
                "input_length": len(sanitized_text),
                "processing_time": processing_time,
                "method": "API-based",
                "response_length": len(response)
            }
            self.analysis_history.append(analysis_record)
            
            return response, processing_time, "API-based Model"
            
        except Exception as e:
            return f"Error in analysis: {str(e)}", 0, "Error"
    
    def generate_secure_summary(self, text):
        """Generate a privacy-aware summary"""
        try:
            sanitized_text = self.sanitize_input(text)
            
            prompt = f"""Create a secure summary of the following text that removes personally identifiable information:

Original Text: "{sanitized_text}"

Create a summary that:
1. Preserves the main message and intent
2. Removes names, addresses, phone numbers, emails
3. Generalizes specific locations to regions
4. Maintains business context without revealing sensitive details

Secure Summary:"""
            
            start_time = time.time()
            
            response = client.text_generation(
                prompt,
                max_new_tokens=150,
                temperature=0.4,
                do_sample=True,
                return_full_text=False
            )
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            return response, processing_time
            
        except Exception as e:
            return f"Error generating summary: {str(e)}", 0
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.analysis_history:
            return "No analysis performed yet."
        
        total_requests = len(self.analysis_history)
        avg_processing_time = sum(record["processing_time"] for record in self.analysis_history) / total_requests
        total_input_chars = sum(record["input_length"] for record in self.analysis_history)
        
        stats = f"""
## Performance Statistics 📊

**Session ID:** {self.session_id}
**Total Analyses:** {total_requests}
**Average Processing Time:** {avg_processing_time:.2f} seconds
**Total Characters Processed:** {total_input_chars:,}
**Model Type:** API-based (Remote Inference)

### Recent Activity:
"""
        
        for record in self.analysis_history[-3:]:  # Show last 3 records
            stats += f"- {record['timestamp']}: {record['processing_time']}s ({record['input_length']} chars)\n"
        
        return stats

# Initialize the analytics engine
analytics = SecureChatAnalytics()

def analyze_text_interface(text, analysis_type):
    """Main interface function for text analysis"""
    if not text.strip():
        return "Please enter some text to analyze.", "", ""
    
    if analysis_type == "Sentiment & Tone Analysis":
        result, time_taken, model_info = analytics.analyze_sentiment_tone(text)
        performance_info = f"⚡ Processed in {time_taken}s using {model_info}"
        stats = analytics.get_performance_stats()
        return result, performance_info, stats
    
    elif analysis_type == "Secure Summary":
        result, time_taken = analytics.generate_secure_summary(text)
        performance_info = f"⚡ Processed in {time_taken}s using API-based Model"
        stats = analytics.get_performance_stats()
        return result, performance_info, stats
    
    else:
        return "Please select an analysis type.", "", ""

def reset_session():
    """Reset the current session"""
    global analytics
    analytics = SecureChatAnalytics()
    return "🔄 New session started!", "", ""

# Create the Gradio interface
with gr.Blocks(
    title="SecureChat Analytics Platform",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
) as demo:
    
    gr.Markdown("""
    # 🔒 SecureChat Analytics Platform
    
    **Privacy-First Text Analysis with VaultGemma**
    
    This platform demonstrates **API-based ML deployment** using Google's VaultGemma model for secure text analysis.
    Features include sentiment analysis, tone detection, and privacy-aware text summarization.
    
    🚀 **Key Features:**
    - Real-time sentiment and tone analysis
    - Privacy-aware text summarization
    - Security risk assessment
    - Performance monitoring and analytics
    - Session-based privacy protection
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Enter Text for Analysis",
                placeholder="Type your message, email, or document content here...",
                lines=5,
                max_lines=10
            )
            
            analysis_type = gr.Dropdown(
                choices=["Sentiment & Tone Analysis", "Secure Summary"],
                label="🔍 Analysis Type",
                value="Sentiment & Tone Analysis"
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🚀 Analyze Text", variant="primary")
                reset_btn = gr.Button("🔄 New Session", variant="secondary")
        
        with gr.Column(scale=3):
            analysis_output = gr.Textbox(
                label="📊 Analysis Results",
                lines=8,
                max_lines=15,
                interactive=False
            )
            
            performance_info = gr.Textbox(
                label="⚡ Performance Info",
                lines=2,
                interactive=False
            )
            
            stats_output = gr.Markdown(
                label="📈 Session Statistics",
                value="No analysis performed yet."
            )
    
    gr.Markdown("""
    ## 🛠️ Technical Details
    
    - **Model:** Google VaultGemma-1b (Privacy-focused language model)
    - **Deployment:** API-based using Hugging Face Inference Client
    - **Security:** Input sanitization, PII removal, session isolation
    - **Performance:** Real-time processing with performance tracking
    
    ## 🔒 Privacy Features
    
    - Session-based isolation with unique IDs
    - Input sanitization to remove harmful content
    - PII-aware summarization
    - No persistent storage of sensitive data
    """)
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_text_interface,
        inputs=[text_input, analysis_type],
        outputs=[analysis_output, performance_info, stats_output]
    )
    
    reset_btn.click(
        fn=reset_session,
        outputs=[performance_info, analysis_output, stats_output]
    )
    
    # Auto-update stats when analysis type changes
    analysis_type.change(
        fn=lambda: analytics.get_performance_stats(),
        outputs=[stats_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )