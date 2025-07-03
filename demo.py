import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInferenceDemo:
    """Simple LLM Inference Demo with LoRA Distillation"""
    
    def __init__(self):
        # Configuration: Add your models and LoRAs here
        self.config = {
            "base_models": {
                "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                "GPT2-XL": "openai-community/gpt2-xl",
            },
            "lora_adapters": {
                "TinyLlama-1.1B": [
                    {"name": "Base Model (No LoRA)", "id": None},
                    {"name": "MC-KD LoRA", "id": "trongg/tinyllama-mckd-dolly"},
                    {"name": "SFT LoRA", "id": "trongg/tinyllamasftdolly"},
                    {"name": "ULD LoRA", "id": "trongg/tinyllamaulddolly"},
                    {"name": "DSKD LoRA", "id": "trongg/tinyllamadskddolly"},
                ],
                "GPT2-XL": [
                    {"name": "Base Model (No LoRA)", "id": None},
                    {"name": "MC-KD LoRA", "id": "trongg/gptxl-mckd-dolly"},
                    {"name": "SFT LoRA", "id": "trongg/gptxlsftdolly"},
                    {"name": "ULD LoRA", "id": "trongg/gptxlulddolly"},
                    {"name": "DSKD LoRA", "id": "trongg/gptxldskddolly"}
                ]
            }
        }
        
        # Model state
        self.current_base_model = None
        self.current_model = None  # This will be PeftModel once first LoRA is loaded
        self.current_tokenizer = None
        self.current_base_model_name = None
        self.current_lora_name = None
        self.loaded_adapters = set()  # Track loaded adapters
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def format_prompt(self, instruction: str, context: str) -> str:
        """Format data into appropriate prompt for the model."""
        prompt = f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
"""
        return prompt
    
    def unload_all_lora(self):
        """Unload all LoRA adapters using PEFT's unload() method"""
        if (self.current_model and 
            hasattr(self.current_model, 'unload') and
            hasattr(self.current_model, 'peft_config') and
            self.current_model.peft_config):
            try:
                self.current_model.unload()
                self.loaded_adapters.clear()
                logger.info("✅ All LoRA adapters unloaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to unload LoRA: {e}")
    
    def delete_adapter(self, adapter_name: str):
        """Delete specific adapter using PEFT's delete_adapter() method"""
        if (self.current_model and 
            hasattr(self.current_model, 'delete_adapter') and
            adapter_name in self.loaded_adapters):
            try:
                self.current_model.delete_adapter(adapter_name)
                self.loaded_adapters.discard(adapter_name)
                logger.info(f"✅ Adapter '{adapter_name}' deleted")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete adapter {adapter_name}: {e}")
    
    def clear_memory(self):
        """Clear GPU memory and unload models"""
        # Unload all LoRA adapters first
        self.unload_all_lora()
        
        # Clear models
        if self.current_model:
            del self.current_model
            self.current_model = None
        if self.current_base_model:
            del self.current_base_model
            self.current_base_model = None
        if self.current_tokenizer:
            del self.current_tokenizer
            self.current_tokenizer = None
            
        # Reset state
        self.current_base_model_name = None
        self.current_lora_name = None
        self.loaded_adapters.clear()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Memory cleared")
    
    def load_base_model(self, model_name: str) -> tuple:
        """Load base model and tokenizer"""
        try:
            logger.info(f"Loading base model: {model_name}")
            model_id = self.config["base_models"][model_name]
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info(f"✅ Base model loaded: {model_name}")
            return model, tokenizer, "✅ Model loaded successfully!"
            
        except Exception as e:
            error_msg = f"❌ Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg
    
    def switch_lora_adapter(self, lora_name: str, lora_id: Optional[str]) -> str:
        """Switch LoRA adapter using PEFT's load_adapter() and set_adapter() methods"""
        try:
            if lora_id is None:
                # No LoRA - unload all adapters to return to base model
                if (self.current_model and 
                    hasattr(self.current_model, 'peft_config') and 
                    self.current_model.peft_config):
                    self.unload_all_lora()
                    logger.info("✅ Unloaded all LoRAs, using base model")
                    return "✅ Using base model (no LoRA)"
                else:
                    # If no PeftModel exists yet, just use base model
                    logger.info("✅ Using base model (no LoRA)")
                    return "✅ Using base model (no LoRA)"
            
            # Validate that this LoRA is for the current base model
            if self.current_base_model_name:
                valid_loras = [adapter["id"] for adapter in self.config["lora_adapters"][self.current_base_model_name]]
                if lora_id not in valid_loras:
                    error_msg = f"❌ LoRA '{lora_id}' is not compatible with base model '{self.current_base_model_name}'"
                    logger.error(error_msg)
                    return error_msg
            
            # Create PeftModel if it doesn't exist yet
            if not hasattr(self.current_model, 'peft_config'):
                logger.info(f"Creating initial PeftModel with adapter: {lora_name}")
                self.current_model = PeftModel.from_pretrained(
                    self.current_base_model,
                    lora_id,
                    adapter_name=lora_name,
                    torch_dtype=torch.float16
                )
                self.loaded_adapters.add(lora_name)
                logger.info(f"✅ Initial LoRA loaded: {lora_name}")
                return f"✅ LoRA loaded: {lora_name}"
            
            # Check if adapter is already loaded
            if lora_name not in self.loaded_adapters:
                logger.info(f"Loading new adapter: {lora_name}")
                self.current_model.load_adapter(lora_id, adapter_name=lora_name)
                self.loaded_adapters.add(lora_name)
                logger.info(f"✅ Adapter loaded: {lora_name}")
            
            # Set as active adapter
            self.current_model.set_adapter(lora_name)
            logger.info(f"✅ Active adapter set to: {lora_name}")
            return f"✅ Switched to LoRA: {lora_name}"
            
        except Exception as e:
            error_msg = f"❌ Failed to switch LoRA {lora_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def switch_model(self, base_model_name: str, lora_name: str) -> str:
        """Switch to different base model and LoRA combination"""
        try:
            # Check if we need to reload base model
            if self.current_base_model_name != base_model_name:
                # IMPORTANT: Clear everything when switching base models
                # to prevent LoRA conflicts between different models
                self.clear_memory()
                
                # Load new base model
                base_model, tokenizer, load_msg = self.load_base_model(base_model_name)
                if base_model is None:
                    return load_msg
                
                self.current_base_model = base_model
                self.current_tokenizer = tokenizer
                self.current_base_model_name = base_model_name
                self.current_model = base_model  # Initialize with base model
                self.current_lora_name = None  # Reset LoRA state
                self.loaded_adapters.clear()  # Clear adapter tracking
                
                logger.info(f"✅ Base model switched to: {base_model_name}")
            
            # Switch LoRA if needed
            if self.current_lora_name != lora_name:
                # Find LoRA ID for the CURRENT base model
                lora_id = None
                for adapter in self.config["lora_adapters"][base_model_name]:
                    if adapter["name"] == lora_name:
                        lora_id = adapter["id"]
                        break
                
                if lora_id is None and lora_name != "Base Model (No LoRA)":
                    return f"❌ LoRA '{lora_name}' not found for {base_model_name}"
                
                # Switch to new LoRA using PEFT methods
                lora_msg = self.switch_lora_adapter(lora_name, lora_id)
                self.current_lora_name = lora_name
                
                status_msg = f"🔄 Switched to: {base_model_name} + {lora_name}\n{lora_msg}"
            else:
                status_msg = f"✅ Already using: {base_model_name} + {lora_name}"
            
            logger.info(status_msg)
            return status_msg
            
        except Exception as e:
            error_msg = f"❌ Failed to switch model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def generate_response(
        self, 
        instruction: str, 
        context: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> str:
        """Generate response using current model"""
        try:
            if self.current_model is None or self.current_tokenizer is None:
                return "❌ No model loaded. Please select a model first."
            
            # Format prompt
            prompt = self.format_prompt(instruction, context)
            
            # Tokenize
            inputs = self.current_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            logger.info(f"Generating with: {self.current_base_model_name} + {self.current_lora_name}")
            
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    eos_token_id=self.current_tokenizer.eos_token_id,
                    pad_token_id=self.current_tokenizer.pad_token_id,
                )
            
            # Decode response
            generated_text = self.current_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            
            # Extract only the response part
            response = generated_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def update_lora_choices(self, base_model_name: str) -> gr.Dropdown:
        """Update LoRA choices based on selected base model"""
        lora_choices = [adapter["name"] for adapter in self.config["lora_adapters"][base_model_name]]
        return gr.Dropdown(choices=lora_choices, value=lora_choices[0])
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="LLM Inference Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 LLM Inference Demo with LoRA Switching")
            gr.Markdown("Switch between different base models and LoRA adapters for text generation.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Model selection
                    gr.Markdown("## 🔧 Model Configuration")
                    
                    base_model_dropdown = gr.Dropdown(
                        choices=list(self.config["base_models"].keys()),
                        value=list(self.config["base_models"].keys())[0],
                        label="Base Model",
                        interactive=True
                    )
                    
                    lora_dropdown = gr.Dropdown(
                        choices=[adapter["name"] for adapter in self.config["lora_adapters"][list(self.config["base_models"].keys())[0]]],
                        value="Base Model (No LoRA)",
                        label="LoRA Adapter",
                        interactive=True
                    )
                    
                    load_button = gr.Button("🔄 Load Model", variant="primary")
                    model_status = gr.Textbox(
                        label="Model Status", 
                        value="No model loaded", 
                        interactive=False
                    )
                    
                    # Generation parameters
                    gr.Markdown("## ⚙️ Generation Parameters")
                    
                    max_tokens = gr.Slider(
                        minimum=10, maximum=512, value=128, step=1,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Temperature"
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=100, value=10, step=1,
                        label="Top K"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        label="Top P"
                    )
                    do_sample = gr.Checkbox(value=True, label="Do Sample")
                
                with gr.Column(scale=2):
                    # Input fields
                    gr.Markdown("## 📝 Input")
                    
                    instruction_input = gr.Textbox(
                        label="Instruction",
                        placeholder="Enter your instruction here...",
                        lines=3
                    )
                    
                    context_input = gr.Textbox(
                        label="Context", 
                        placeholder="Enter context information here...",
                        lines=3
                    )
                    
                    generate_button = gr.Button("🚀 Generate", variant="primary", size="lg")
                    
                    # Output
                    gr.Markdown("## 📤 Generated Response")
                    output_text = gr.Textbox(
                        label="Response",
                        lines=10,
                        interactive=False
                    )
                    
                    # Example inputs
                    gr.Markdown("## 💡 Example Inputs")
                    examples = gr.Examples(
                        examples=[
                            ["What is the largest city on the Mississippi River?", "Memphis is the fifth-most populous city in the Southeast, the nation's 28th-largest overall, as well as the largest city bordering the Mississippi River and third largest Metropolitan statistical area behind Saint Louis, MO and the Twin Cities on the Mississippi River. The Memphis metropolitan area includes West Tennessee and the greater Mid-South region, which includes portions of neighboring Arkansas, Mississippi and the Missouri Bootheel. One of the more historic and culturally significant cities of the Southern United States, Memphis has a wide variety of landscapes and distinct neighborhoods."],
                            ["From the passage provided, extract the second studio album that Taylor Swift released.", "Swift signed a record deal with Big Machine Records in 2005 and released her eponymous debut album the following year. With 157 weeks on the Billboard 200 by December 2009, the album was the longest-charting album of the 2000s decade. Swift's second studio album, Fearless (2008), topped the Billboard 200 for 11 weeks and was the only album from the 2000s decade to spend one year in the top 10. The album was certified Diamond by the RIAA. It also topped charts in Australia and Canada, and has sold 12 million copies worldwide. Her third studio album, the self-written Speak Now (2010), spent six weeks atop the Billboard 200 and topped charts in Australia, Canada, and New Zealand."],
                        ],
                        inputs=[instruction_input, context_input]
                    )
            
            # Event handlers
            base_model_dropdown.change(
                fn=self.update_lora_choices,
                inputs=[base_model_dropdown],
                outputs=[lora_dropdown]
            )
            
            load_button.click(
                fn=self.switch_model,
                inputs=[base_model_dropdown, lora_dropdown],
                outputs=[model_status]
            )
            
            generate_button.click(
                fn=self.generate_response,
                inputs=[
                    instruction_input,
                    context_input,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    do_sample
                ],
                outputs=[output_text]
            )
        
        return demo

# Launch the demo
def main():
    """Main function to launch the demo"""
    
    # Initialize demo
    demo_app = LLMInferenceDemo()
    
    # Create interface
    interface = demo_app.create_interface()
    
    # Launch
    interface.launch(
        share=False,        # Set to True if you want public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,   # Default Gradio port
        show_error=True,    # Show detailed errors
        quiet=False         # Show startup logs
    )

if __name__ == "__main__":
    main()