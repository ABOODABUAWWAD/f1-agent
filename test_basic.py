#!/usr/bin/env python3
"""
Basic test to check if our system is working
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test all critical imports"""
    print("=== Testing Imports ===")
    
    try:
        import transformers
        print("✅ transformers imported")
    except Exception as e:
        print(f"❌ transformers failed: {e}")
    
    try:
        import chromadb
        print("✅ chromadb imported")
    except Exception as e:
        print(f"❌ chromadb failed: {e}")
    
    try:
        import sentence_transformers
        print("✅ sentence_transformers imported")
    except Exception as e:
        print(f"❌ sentence_transformers failed: {e}")
    
    try:
        from src.infra.vector_store import add_context, query_context
        print("✅ vector_store imported")
    except Exception as e:
        print(f"❌ vector_store failed: {e}")
    
    try:
        from src.agent.model_loader import load_local_model
        print("✅ model_loader imported")
    except Exception as e:
        print(f"❌ model_loader failed: {e}")

def test_vector_store():
    """Test ChromaDB functionality"""
    print("\n=== Testing ChromaDB ===")
    
    try:
        from src.infra.vector_store import add_context, query_context, get_collection_stats
        
        # Add test data
        add_context("Test: I love machine learning", {"test": True, "source": "basic_test"})
        print("✅ Added test context")
        
        # Query test data
        results = query_context("machine learning", top_k=3)
        print(f"✅ Query results: {len(results.get('documents', [[]])[0])} documents found")
        
        # Get stats
        stats = get_collection_stats()
        print(f"✅ Collection stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_model():
    """Test local model loading"""
    print("\n=== Testing Local Model ===")
    
    try:
        from src.agent.model_loader import load_local_model, generate_local
        
        # Try to load model
        tokenizer, model = load_local_model()
        print("✅ Local model loaded (or mock loaded)")
        
        # Test generation
        if hasattr(model, 'generate') and callable(model.generate):
            response = generate_local(tokenizer, model, "Hello, how are you?", max_new_tokens=50)
            print(f"✅ Generated response: {response[:100]}...")
        else:
            print("✅ Mock model loaded for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Local model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent():
    """Test the full agent workflow"""
    print("\n=== Testing Agent Workflow ===")
    
    try:
        from src.agent.agent import run_agent
        
        # Test basic conversation
        result = run_agent("Hello, my name is Alice", "test_user")
        print(f"✅ Agent response: {result['response'][:100]}...")
        print(f"   Model used: {result['model_used']}")
        print(f"   Processing time: {result['processing_time']}s")
        print(f"   Context items: {result['context_items']}")
        
        # Test memory functionality
        result2 = run_agent("Remember: I love pizza", "test_user")
        print(f"✅ Memory test: {result2['response'][:100]}...")
        print(f"   Memory saved: {result2['memory_saved']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Anigma F1 System Tests...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run all tests
    test_imports()
    
    vector_ok = test_vector_store()
    model_ok = test_local_model()
    agent_ok = test_agent()
    
    print("\n=== SUMMARY ===")
    print(f"ChromaDB: {'✅' if vector_ok else '❌'}")
    print(f"Local Model: {'✅' if model_ok else '❌'}")
    print(f"Agent: {'✅' if agent_ok else '❌'}")
    
    if vector_ok and model_ok and agent_ok:
        print("🎉 All core systems working!")
    else:
        print("⚠️  Some systems need attention")