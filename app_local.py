import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import time
import re
from datetime import datetime
import hashlib
import psutil
import os

class LocalSecureChatAnalytics:
    def __init__(self):
        self.session_id = self.generate_session_id()
        self.analysis_history = []
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def generate_session_id(self):
        """Generate a unique session ID for privacy tracking"""
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    def load_model(self):
        """Load the VaultGemma model locally"""
        if self.model_loaded:
            return "Model already loaded!"
        
        try:
            print("Loading VaultGemma model locally...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/vaultgemma-1b",
                token="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"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/vaultgemma-1b",
                token="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",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create pipeline for easier text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model_loaded = True
            return f"✅ Model loaded successfully on {self.device.upper()}!"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def get_system_stats(self):
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_info = f"GPU Memory: {gpu_memory:.2f}GB / {gpu_total:.2f}GB"
        else:
            gpu_info = "GPU: Not available"
        
        return f"""
**System Resources:**
- CPU Usage: {cpu_percent}%
- RAM Usage: {memory.percent}% ({memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB)
- {gpu_info}
- Device: {self.device.upper()}
"""
    
    def sanitize_input(self, text):
        """Basic input sanitization for security"""
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # Keep only safe characters
        return text.strip()
    
    def analyze_with_local_model(self, text, analysis_type):
        """Analyze text using locally loaded model"""
        if not self.model_loaded:
            return "Please load the model first!", 0
        
        try:
            sanitized_text = self.sanitize_input(text)
            
            if analysis_type == "Sentiment & Tone Analysis":
                prompt = f"""Analyze the following text for sentiment, tone, and security:

Text: "{sanitized_text}"

Analysis:
- Sentiment: [Positive/Negative/Neutral]
- Tone: [Professional/Casual/Friendly/etc.]
- Security Risk: [Low/Medium/High]
- Privacy Score: [1-10]

Response:"""

            elif analysis_type == "Secure Summary":
                prompt = f"""Create a secure summary removing PII:

Text: "{sanitized_text}"

Secure Summary:"""

            elif analysis_type == "Content Classification":
                prompt = f"""Classify the content type and safety:

Text: "{sanitized_text}"

Classification:
- Content Type: [Business/Personal/Technical/etc.]
- Safety Level: [Safe/Caution/Risk]
- Audience: [Public/Internal/Confidential]

Result:"""
            
            start_time = time.time()
            
            # Generate response using local model
            response = self.pipeline(
                prompt,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Extract generated text
            generated_text = response[0]['generated_text'] if response else "No response generated"
            
            # Store analysis in history
            analysis_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": self.session_id,
                "input_length": len(sanitized_text),
                "processing_time": processing_time,
                "method": "Local Model",
                "analysis_type": analysis_type,
                "device": self.device
            }
            self.analysis_history.append(analysis_record)
            
            return generated_text, processing_time
            
        except Exception as e:
            return f"Error in analysis: {str(e)}", 0
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        if not self.analysis_history:
            return "No analysis performed yet."
        
        total_requests = len(self.analysis_history)
        avg_processing_time = sum(record["processing_time"] for record in self.analysis_history) / total_requests
        total_input_chars = sum(record["input_length"] for record in self.analysis_history)
        
        # Analysis type breakdown
        type_counts = {}
        for record in self.analysis_history:
            analysis_type = record.get("analysis_type", "Unknown")
            type_counts[analysis_type] = type_counts.get(analysis_type, 0) + 1
        
        stats = f"""
## 📊 Local Model Performance Statistics

**Session ID:** `{self.session_id}`
**Model Status:** {'✅ Loaded' if self.model_loaded else '❌ Not Loaded'}
**Total Analyses:** {total_requests}
**Average Processing Time:** {avg_processing_time:.2f} seconds
**Total Characters Processed:** {total_input_chars:,}
**Deployment Type:** Local Model Execution
**Device:** {self.device.upper()}

### Analysis Type Breakdown:
"""
        
        for analysis_type, count in type_counts.items():
            stats += f"- **{analysis_type}:** {count} requests\n"
        
        stats += "\n### Recent Activity:\n"
        for record in self.analysis_history[-3:]:  # Show last 3 records
            stats += f"- `{record['timestamp']}`: {record['analysis_type']} ({record['processing_time']}s)\n"
        
        return stats

# Initialize the local analytics engine
local_analytics = LocalSecureChatAnalytics()

def load_model_interface():
    """Interface function to load the model"""
    result = local_analytics.load_model()
    stats = local_analytics.get_performance_stats()
    system_stats = local_analytics.get_system_stats()
    return result, stats, system_stats

def analyze_local_interface(text, analysis_type):
    """Main interface function for local model analysis"""
    if not text.strip():
        return "Please enter some text to analyze.", "", ""
    
    if not local_analytics.model_loaded:
        return "⚠️ Please load the model first using the 'Load Model' button.", "", ""
    
    result, time_taken = local_analytics.analyze_with_local_model(text, analysis_type)
    performance_info = f"⚡ Processed in {time_taken}s using Local VaultGemma Model on {local_analytics.device.upper()}"
    stats = local_analytics.get_performance_stats()
    
    return result, performance_info, stats

def get_system_info():
    """Get current system information"""
    return local_analytics.get_system_stats()

def reset_local_session():
    """Reset the current session"""
    global local_analytics
    old_model = local_analytics.model
    old_tokenizer = local_analytics.tokenizer
    old_pipeline = local_analytics.pipeline
    old_loaded = local_analytics.model_loaded
    
    local_analytics = LocalSecureChatAnalytics()
    
    # Preserve loaded model
    if old_loaded:
        local_analytics.model = old_model
        local_analytics.tokenizer = old_tokenizer
        local_analytics.pipeline = old_pipeline
        local_analytics.model_loaded = old_loaded
    
    return "🔄 New session started (model preserved)!", "", ""

# Create the Gradio interface for local model
with gr.Blocks(
    title="SecureChat Analytics - Local Model",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald")
) as demo:
    
    gr.Markdown("""
    # 🖥️ SecureChat Analytics Platform - Local Model Edition
    
    **Privacy-First Text Analysis with Local VaultGemma Deployment**
    
    This platform demonstrates **local ML model deployment** using Google's VaultGemma model running on your hardware.
    Experience the benefits of local processing: complete privacy, no network dependency, and full control.
    
    🏠 **Local Deployment Benefits:**
    - Complete data privacy (no data leaves your system)
    - No network dependency for inference
    - Consistent performance (no API rate limits)
    - Full control over model parameters
    - Cost-effective for high-volume usage
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            load_model_btn = gr.Button("🚀 Load VaultGemma Model", variant="primary")
            model_status = gr.Textbox(
                label="Model Status",
                value="Model not loaded. Click 'Load Model' to start.",
                interactive=False,
                lines=2
            )
            
            system_info = gr.Markdown(
                value="Click 'Refresh System Info' to see resource usage."
            )
            
            refresh_info_btn = gr.Button("🔄 Refresh System Info", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Enter Text for Local Analysis",
                placeholder="Type your message, email, or document content here...",
                lines=5,
                max_lines=10
            )
            
            analysis_type = gr.Dropdown(
                choices=["Sentiment & Tone Analysis", "Secure Summary", "Content Classification"],
                label="🔍 Analysis Type",
                value="Sentiment & Tone Analysis"
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze with Local Model", variant="primary")
                reset_btn = gr.Button("🔄 New Session", variant="secondary")
        
        with gr.Column(scale=3):
            analysis_output = gr.Textbox(
                label="📊 Local Analysis Results",
                lines=8,
                max_lines=15,
                interactive=False
            )
            
            performance_info = gr.Textbox(
                label="⚡ Local Performance Info",
                lines=2,
                interactive=False
            )
            
            stats_output = gr.Markdown(
                value="Load the model and perform analysis to see statistics."
            )
    
    gr.Markdown("""
    ## 🔧 Local Deployment Technical Details
    
    - **Model:** Google VaultGemma-1b loaded locally
    - **Hardware:** Automatic CPU/GPU detection and optimization
    - **Memory Management:** Efficient model loading with torch optimizations
    - **Performance:** Real-time processing with resource monitoring
    - **Privacy:** All processing happens locally, no external API calls
    
    ## 📈 Performance Comparison
    
    **Local vs API-based deployment trade-offs:**
    - **Latency:** Local = Consistent, API = Variable (network dependent)
    - **Privacy:** Local = Complete, API = Dependent on provider
    - **Cost:** Local = Hardware investment, API = Per-request pricing
    - **Scalability:** Local = Limited by hardware, API = Highly scalable
    - **Maintenance:** Local = Self-managed, API = Provider-managed
    """)
    
    # Event handlers
    load_model_btn.click(
        fn=load_model_interface,
        outputs=[model_status, stats_output, system_info]
    )
    
    analyze_btn.click(
        fn=analyze_local_interface,
        inputs=[text_input, analysis_type],
        outputs=[analysis_output, performance_info, stats_output]
    )
    
    reset_btn.click(
        fn=reset_local_session,
        outputs=[performance_info, analysis_output, stats_output]
    )
    
    refresh_info_btn.click(
        fn=get_system_info,
        outputs=[system_info]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True
    )