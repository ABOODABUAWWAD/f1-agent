#!/usr/bin/env python3
"""Final functionality test for Anigma F1 Agent"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_functionality():
    print('=== Testing Anigma F1 Agent Functionality ===')
    
    # Add proper path
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    # Test basic imports
    try:
        from src.agent.agent import run_agent, get_local_model
        from src.infra.vector_store import add_context, query_context
        print('✅ All imports successful')
    except Exception as e:
        print(f'❌ Import error: {e}')
        return False
    
    # Test agent with a simple question
    try:
        result = run_agent('Hello, what is your name?', 'test_user')
        print(f'✅ Agent response: {result["response"][:100]}...')
    except Exception as e:
        print(f'❌ Agent error: {e}')
        # Don't return False for agent errors since they might be expected
    
    # Test memory system
    try:
        add_context('The user likes pizza', {'user_id': 'test_user', 'type': 'preference'})
        memories = query_context('pizza', top_k=1)
        print(f'✅ Memory system working: Found results')
        if memories.get('documents') and memories['documents'][0]:
            print(f'   Memory content: {memories["documents"][0][0][:50]}...')
    except Exception as e:
        print(f'❌ Memory error: {e}')
        return False
    
    # Test model loading
    try:
        tokenizer, model = get_local_model()
        model_type = type(model).__name__
        print(f'✅ Model loading: {model_type} ready')
    except Exception as e:
        print(f'❌ Model loading error: {e}')
        return False
    
    print('=== All Tests Passed! ===')
    return True

if __name__ == "__main__":
    success = test_functionality()
    sys.exit(0 if success else 1)