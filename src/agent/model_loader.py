"""import and manage local model loading and inference"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Local model options - now that we have HF auth, we can use better models
# Good conversational model, manageable size
LOCAL_MODEL_ID = "microsoft/DialoGPT-medium"
# Alternative production options:
# "mistralai/Mistral-7B-Instruct-v0.1"  # Better but larger
# "Qwen/Qwen2.5-7B-Instruct"           # Even better but requires more VRAM
# "distilgpt2"                          # Fallback for testing


def load_local_model(model_id=None, cache_dir="./models"):
    """Load local quantized model for RTX 3050"""
    model_id = model_id or LOCAL_MODEL_ID
    print(f"Attempting to load local model: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure quantization for RTX 3050
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            dtype=torch.float16,  # Updated parameter name
            quantization_config=quantization_config,
            trust_remote_code=True
        )

        print(f"✅ Successfully loaded {model_id}")
        return tokenizer, model

    except Exception as e:
        print(f"❌ Failed to load {model_id}: {e}")
        print("🔄 Falling back to mock local model for demonstration...")
        return MockTokenizer(), MockModel()


def generate_local(tokenizer, model, prompt, max_new_tokens=256):
    """Generate response using local model"""
    if isinstance(model, MockModel):
        return model.generate(prompt, max_new_tokens)

    # For DialoGPT, we need to format the conversation properly
    # DialoGPT expects: conversation history + eos_token + user_input + eos_token
    conversation_text = prompt + tokenizer.eos_token
    
    # Use tokenizer with proper attention mask handling
    inputs = tokenizer(
        conversation_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        return_attention_mask=True
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Move to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens (skip the input)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Clean up the response
    response = response.strip()
    
    # If response is empty or just tokens, provide a fallback
    if not response or len(response.strip()) < 3:
        return "I understand you're asking something, but I'm having trouble generating a proper response. Could you please rephrase your question?"
    
    return response


class MockTokenizer:
    """Mock tokenizer for demonstration"""
    def __init__(self):
        self.eos_token = "<|endoftext|>"
        self.pad_token = self.eos_token


class MockModel:
    """Mock model for demonstration when real models can't be loaded"""
    def generate(self, prompt, max_new_tokens=256):
        # Simple rule-based responses for demonstration
        prompt_lower = prompt.lower()

        if "hello" in prompt_lower or "hi" in prompt_lower:
            return ("Hello! I'm your AI assistant. I can help you with "
                    "questions, remember information, and more. Try asking "
                    "me something or say 'Remember: [some fact]' to store "
                    "information!")

        elif "who are you" in prompt_lower:
            return ("I'm Anigma F1, your local AI assistant! I use a "
                    "hybrid approach - a fast local model for quick "
                    "responses and a powerful remote model for complex "
                    "tasks. I also have persistent memory to remember "
                    "our conversations.")

        elif "remember:" in prompt_lower:
            return ("I've saved that information to my memory! You can "
                    "ask me about it later and I'll remember.")

        elif "help" in prompt_lower:
            return ("I can help you with various tasks:\n"
                    "• Answer questions\n"
                    "• Remember information (use 'Remember: [fact]')\n"
                    "• Have conversations\n"
                    "• Use complex reasoning (I'll automatically switch to "
                    "my powerful remote model when needed)\n\n"
                    "What would you like to know?")

        elif "weather" in prompt_lower:
            return ("I don't have access to real-time weather data, but I "
                    "can help you with other information! If you need "
                    "weather updates, I'd recommend checking a weather "
                    "service or app.")

        elif ("quantum" in prompt_lower or "analyze" in prompt_lower or
              "explain in detail" in prompt_lower):
            return ("[This would use the remote Qwen2.5-7B model for "
                    "complex queries, but requires HF_TOKEN setup]")

        else:
            return (f"I understand you're asking about: "
                    f"'{prompt.strip()}'. This is a demonstration with a "
                    f"mock local model. To use real AI models, please set "
                    f"up your Hugging Face token with: huggingface-cli login")


__all__ = [
    "load_local_model",
    "generate_local",
    "MockTokenizer",
    "MockModel"
]
