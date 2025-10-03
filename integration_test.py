#!/usr/bin/env python3
"""
Start server and run integration tests
"""

import sys
import os
import time
import threading
import requests
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


def test_server_endpoints():
    """Test server endpoints"""
    print("=== Testing Server Endpoints ===")
    
    # Wait for server to start
    time.sleep(3)
    
    base_url = "http://localhost:8000"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        result = response.json()
        print(f"✅ Health Check: {result}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return
    
    # Test chat endpoint with local model query
    try:
        test_request = {
            "user_id": "test_user",
            "text": "Hello, what can you help me with?"
        }
        response = requests.post(f"{base_url}/ask", json=test_request, timeout=30)
        result = response.json()
        print(f"✅ Chat Response: {result['reply'][:100]}...")
        print(f"   Model: {result['model_used']} | Time: {result['processing_time']:.2f}s")
    except Exception as e:
        print(f"❌ Chat Test Failed: {e}")
    
    # Test memory endpoint
    try:
        memory_request = {
            "text": "Test memory: User prefers coffee over tea",
            "metadata": {"test": True}
        }
        response = requests.post(f"{base_url}/memory/add", json=memory_request, timeout=10)
        result = response.json()
        print(f"✅ Memory Add: {result}")
    except Exception as e:
        print(f"❌ Memory Test Failed: {e}")

if __name__ == "__main__":
    print("🔧 Integration Test: Server + Endpoints")
    print("=" * 50)
    
    
    # Run tests
    test_server_endpoints()
    
    print("=" * 50)
    print("🎉 Integration test completed!")
    print("💡 Server is still running on http://localhost:8001")
    print("📖 Try visiting http://localhost:8001/docs for API documentation")
    
    # Keep main thread alive briefly
    time.sleep(2)