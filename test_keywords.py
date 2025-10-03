#!/usr/bin/env python3
"""
Direct test of the keyword logic
"""

def test_keywords_directly():
    """Test the keyword logic directly"""
    
    # Copy the exact logic from the file
    def should_use_remote_test(user_text, context_length=0):
        if "use_remote:true" in user_text.lower():
            return True
        
        complex_keywords = [
            "analyze", "explain in detail", "explain", "comprehensive", "research", 
            "compare", "critique", "elaborate", "technical", "complex",
            "quantum", "mechanics", "physics", "mathematics", "mathematical", "algorithm",
            "economic", "implications", "detailed", "thorough", "in-depth", "foundation"
        ]
        
        if any(keyword in user_text.lower() for keyword in complex_keywords):
            return True
        
        total_length = len(user_text.split()) + context_length
        if total_length > 200:
            return True
        
        return False
    
    test_cases = [
        "Explain quantum mechanics in detail",
        "Tell me about quantum computing", 
        "What are the mathematical foundations?",
        "Analyze the economic implications",
        "Hello world"
    ]
    
    for query in test_cases:
        result = should_use_remote_test(query, 0)
        print(f"Query: '{query}' → {'Remote' if result else 'Local'}")
        
        # Debug which keywords match
        keywords = [
            "analyze", "explain in detail", "explain", "comprehensive", "research", 
            "compare", "critique", "elaborate", "technical", "complex",
            "quantum", "mechanics", "physics", "mathematics", "mathematical", "algorithm",
            "economic", "implications", "detailed", "thorough", "in-depth", "foundation"
        ]
        matches = [k for k in keywords if k in query.lower()]
        if matches:
            print(f"  Matched keywords: {matches}")
        else:
            print(f"  No keywords matched")
        print()

if __name__ == "__main__":
    test_keywords_directly()