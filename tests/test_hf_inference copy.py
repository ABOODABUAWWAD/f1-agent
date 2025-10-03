#!/usr/bin/env python3
"""
Test Hugging Face Inference API connection
"""
import os
from huggingface_hub import InferenceClient

def test_hf_inference():
    """Test HF Inference API"""
    # Check if token is available
    token = os.getenv("HF_TOKEN")
    print(f"HF_TOKEN from env: {'Set' if token else 'Not set'}")
    
    # Try to read from HF cache
    try:
        with open("/home/aboodabuawwad/.cache/huggingface/token", "r") as f:
            cached_token = f.read().strip()
            print(f"Cached token: {'Found' if cached_token else 'Empty'}")
            if not token:
                token = cached_token
    except FileNotFoundError:
        print("No cached token found")
    
    if not token:
        print("❌ No token available")
        return
    
    print("✅ Token available, testing inference...")
    
    # Test with a simple model first
    client = InferenceClient(token=token)
    
    # Try a simple completion
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    print("Testing with Qwen model...")
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=50
        )
        print(f"✅ Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Qwen2.5-7B failed: {e}")
        
        # Try alternative model
        print("Trying Qwen1.5 model...")
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen1.5-7B-Chat",
                messages=messages,
                max_tokens=50
            )
            print(f"✅ Alternative success: {response.choices[0].message.content}")
        except Exception as e2:
            print(f"❌ Alternative failed: {e2}")

if __name__ == "__main__":
    test_hf_inference()