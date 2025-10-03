#!/usr/bin/env python3
"""
Test complex queries to verify remote model routing
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.agent.agent import run_agent

def test_complex_queries():
    """Test that complex queries route to remote model"""
    print("=== Testing Complex Query Routing ===")
    
    test_queries = [
        "Explain quantum mechanics in detail",
        "Analyze the economic implications of artificial intelligence", 
        "What are the mathematical foundations of machine learning?",
        "Compare different approaches to neural network architecture",
        "use_remote:true Tell me about your capabilities"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        try:
            result = run_agent(query, "test_user")
            model_used = result['model_used']
            expected = "remote" if any(keyword in query.lower() for keyword in [
                "quantum", "economic", "mathematical", "compare", "use_remote"
            ]) else "local"
            
            status = "✅" if model_used == expected else "❌"
            print(f"{status} Model used: {model_used} (expected: {expected})")
            print(f"   Response: {result['response'][:150]}...")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_complex_queries()