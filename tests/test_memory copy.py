#!/usr/bin/env python3
"""
Test memory/vector store functionality
"""


def test_memory():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.infra.vector_store import add_context, query_context, clear_test_data, get_collection_stats
    
    print("Testing memory functionality...")
    
    # Clean up any existing test data first
    print("Cleaning up existing test data...")
    clear_test_data()
    
    # Show initial stats
    stats = get_collection_stats()
    print(f"Memory store stats: {stats}")
    
    # Add some test data
    print("Adding fresh test data...")
    add_context("I am a software developer", {"test": True, "category": "profession"})
    add_context("I work on AI projects in Python", {"test": True, "category": "skills"})
    add_context("I love quantum computing", {"test": True, "category": "interests"})
    
    # Query the data
    print("\nQuerying for 'software developer'...")
    results = query_context("software developer", top_k=3)
    print("Results:", results)
    
    print("\nQuerying for 'programming background'...")
    results = query_context("programming background", top_k=3)
    print("Results:", results)
    
    print("\nQuerying for 'interests'...")
    results = query_context("interests", top_k=3)
    print("Results:", results)
    
    # Clean up after test
    print("\nCleaning up test data...")
    clear_test_data()
    
    final_stats = get_collection_stats()
    print(f"Final memory store stats: {final_stats}")

if __name__ == "__main__":
    test_memory()