#!/usr/bin/env python3
"""
Test the improvements we made to the system
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_improved_model_decision():
    """Test the improved model decision logic"""
    print("=== Testing Improved Model Decision Logic ===")
    
    from src.agent.remote_qwen_tool import should_use_remote
    
    test_cases = [
        ("Hello", False, "Simple greeting"),
        ("What is 2+2?", False, "Simple math"),
        ("Explain quantum mechanics in detail", True, "Complex physics explanation"),
        ("Analyze the economic implications of AI", True, "Complex analysis"),
        ("Tell me about quantum computing", True, "Contains 'quantum' keyword"),
        ("What are the mathematical foundations?", True, "Contains 'mathematical' keyword"),
    ]
    
    for query, expected, description in test_cases:
        result = should_use_remote(query, 0)
        status = "✅" if result == expected else "❌"
        model = "Remote" if result else "Local" 
        print(f"{status} '{query}' → {model} ({description})")

def test_improved_agent_responses():
    """Test the improved agent response generation"""
    print("\n=== Testing Improved Agent Responses ===")
    
    from src.agent.agent import run_agent
    
    test_cases = [
        "Hello, my name is Alice",
        "What do you know about programming?",
        "Remember: I love machine learning",
        "Tell me about my interests",
    ]
    
    for query in test_cases:
        print(f"\n--- Testing: {query} ---")
        try:
            result = run_agent(query, "test_user")
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"   Model: {result['model_used']} | Time: {result['processing_time']:.2f}s")
            print(f"   Context items: {result['context_items']} | Memory saved: {result['memory_saved']}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_vector_store_functionality():
    """Test vector store is working correctly"""
    print("\n=== Testing Vector Store Functionality ===")
    
    from src.infra.vector_store import add_context, query_context, get_collection_stats
    
    # Add some test data
    test_memories = [
        ("I am a software engineer", {"test": True, "category": "profession"}),
        ("I love Python programming", {"test": True, "category": "interests"}),
        ("I work on AI projects", {"test": True, "category": "work"}),
    ]
    
    for memory, metadata in test_memories:
        add_context(memory, metadata)
        print(f"✅ Added: {memory}")
    
    # Test queries
    test_queries = [
        "programming",
        "software engineer", 
        "AI projects"
    ]
    
    for query in test_queries:
        results = query_context(query, top_k=3)
        doc_count = len(results.get('documents', [[]])[0])
        print(f"✅ Query '{query}' → {doc_count} results")
        if doc_count > 0:
            print(f"   Best match: {results['documents'][0][0][:100]}...")
    
    # Get stats
    stats = get_collection_stats()
    print(f"✅ Collection stats: {stats}")

if __name__ == "__main__":
    print("🔧 Testing Anigma F1 Improvements...")
    print("=" * 50)
    
    test_improved_model_decision()
    test_vector_store_functionality()
    test_improved_agent_responses()
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed! Check results above.")