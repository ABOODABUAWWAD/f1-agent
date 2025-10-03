#!/usr/bin/env python3
"""
Complete test suite for Anigma F1 AI Agent
Runs all tests with proper cleanup and error handling
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test_safe(test_func, test_name):
    """Run a test function safely with error handling"""
    print(f"\n{'='*20} {test_name} {'='*20}")
    try:
        start_time = time.time()
        result = test_func()
        duration = time.time() - start_time
        print(f"✅ {test_name} completed in {duration:.2f}s")
        return True, result
    except Exception as e:
        print(f"❌ {test_name} failed: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False, str(e)

def test_imports():
    """Test all imports"""
    import fastapi
    import uvicorn
    import chromadb
    from sentence_transformers import SentenceTransformer
    import transformers
    print("✅ All imports successful")
    return True

def test_vector_store():
    """Test vector store with cleanup"""
    from src.infra.vector_store import add_context, query_context, clear_test_data, get_collection_stats
    
    # Clean start
    clear_test_data()
    
    # Add test data
    test_data = [
        ("Alice is a Python developer", {"test": True, "user": "alice"}),
        ("Bob loves machine learning", {"test": True, "user": "bob"}),
        ("Charlie studies quantum physics", {"test": True, "user": "charlie"})
    ]
    
    for text, metadata in test_data:
        add_context(text, metadata)
    
    # Test queries
    results = query_context("developer", top_k=2)
    assert results['documents'][0], "Query should return results"
    
    # Cleanup
    clear_test_data()
    print("✅ Vector store test completed with cleanup")
    return True

def test_local_model():
    """Test local model loading and generation"""
    from src.agent.model_loader import load_local_model, generate_local
    
    tokenizer, model = load_local_model()
    response = generate_local(tokenizer, model, "Hello", max_new_tokens=50)
    
    assert response and len(response.strip()) > 0, "Model should generate response"
    print(f"✅ Local model generated: {response[:50]}...")
    return True

def test_agent_workflow():
    """Test the complete agent workflow"""
    from src.agent.agent import run_agent
    
    # Test simple query
    result = run_agent("Hello, my name is TestUser", "test_user_workflow")
    
    assert "response" in result, "Agent should return response"
    assert "model_used" in result, "Agent should indicate model used"
    assert "processing_time" in result, "Agent should report timing"
    
    print(f"✅ Agent workflow: {result['response'][:50]}...")
    return True

def test_api_endpoints():
    """Test API endpoints if server is running"""
    import requests
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running and healthy")
            
            # Test chat endpoint
            chat_data = {"user_id": "test_api", "text": "Hello API"}
            response = requests.post(f"{base_url}/ask", json=chat_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Chat API working: {result['reply'][:30]}...")
                return True
            else:
                print(f"⚠️  Chat API returned {response.status_code}")
                return False
        else:
            print("⚠️  API server not responding properly")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running - skipping API tests")
        return None  # Skip, not a failure
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def test_memory_integration():
    """Test memory system integration"""
    from src.agent.agent import run_agent
    from src.infra.vector_store import clear_test_data
    
    # Clean start
    clear_test_data()
    
    # Add some memory
    result1 = run_agent("Remember: I like pizza", "test_memory_user")
    
    # Query the memory
    result2 = run_agent("What do you know about my food preferences?", "test_memory_user")
    
    # Cleanup
    clear_test_data()
    
    print(f"✅ Memory integration test completed")
    return True

def main():
    """Run all tests"""
    print("🚀 Anigma F1 AI Agent - Complete Test Suite")
    print("=" * 60)
    
    tests = [
        (test_imports, "Import Tests"),
        (test_vector_store, "Vector Store Tests"),
        (test_local_model, "Local Model Tests"),
        (test_agent_workflow, "Agent Workflow Tests"),
        (test_memory_integration, "Memory Integration Tests"),
        (test_api_endpoints, "API Endpoint Tests"),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func, test_name in tests:
        success, result = run_test_safe(test_func, test_name)
        if success is True:
            passed += 1
        elif success is False:
            failed += 1
        else:  # None = skipped
            skipped += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "N/A")
    
    if failed == 0:
        print("\n🎉 All tests passed! System is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())