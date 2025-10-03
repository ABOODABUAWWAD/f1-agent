import os
import requests
from huggingface_hub import InferenceClient
from typing import List, Dict

# Remote heavy model: Qwen2.5-7B via HF Inference API (working model)
REMOTE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Alternative: "Qwen/Qwen1.5-7B-Chat" (also works)

def get_hf_token():
    """Get Hugging Face token from environment"""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set. Please login with: huggingface-cli login")
    return token

def qwen3_infer(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Call Qwen3-Omni-30B via Hugging Face Inference API
    
    Args:
        messages: List of {"role": "user/assistant", "content": "text"} 
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated response text
    """
    try:
        client = InferenceClient(token=get_hf_token())
        
        response = client.chat.completions.create(
            model=REMOTE_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Remote inference error: {e}")
        # Fallback to direct API call if InferenceClient fails
        return qwen3_infer_direct(messages, max_tokens, temperature)

def qwen3_infer_direct(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Direct API call to HF Inference API as fallback
    """
    try:
        headers = {
            "Authorization": f"Bearer {get_hf_token()}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": REMOTE_MODEL_ID,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/" + REMOTE_MODEL_ID,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            elif "generated_text" in result:
                return result["generated_text"].strip()
        
        print(f"Remote API error: {response.status_code} - {response.text}")
        return "Error: Remote model unavailable"
        
    except Exception as e:
        print(f"Direct API call error: {e}")
        return "Error: Remote model failed"

def should_use_remote(user_text: str, context_length: int = 0) -> bool:
    """
    Decision logic: when to use remote vs local model
    
    Args:
        user_text: User's input text
        context_length: Length of retrieved context
    
    Returns:
        True if should use remote heavy model
    """
    # Explicit remote request
    if "use_remote:true" in user_text.lower():
        return True
    
    # Complex reasoning keywords
    complex_keywords = [
        "analyze", "explain in detail", "comprehensive", "research", 
        "compare", "critique", "elaborate", "technical", "complex"
    ]
    
    if any(keyword in user_text.lower() for keyword in complex_keywords):
        return True
    
    # Long context or long query
    total_length = len(user_text.split()) + context_length
    if total_length > 200:  # Token threshold
        return True
    
    return False

# Example usage and testing
if __name__ == "__main__":
    # Test remote inference
    test_messages = [
        {"role": "user", "content": "Hello, can you explain quantum computing in detail?"}
    ]
    
    try:
        response = qwen3_infer(test_messages, max_tokens=100)
        print("Remote model response:", response)
    except Exception as e:
        print("Test failed:", e)