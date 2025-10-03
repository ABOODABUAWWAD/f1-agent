import time
from typing import Dict, Any, List
from src.infra.vector_store import query_context, add_context
from src.agent.model_loader import load_local_model, generate_local
from src.agent.remote_qwen_tool import qwen3_infer, should_use_remote


class AgentState:
    """Simple state management for agent"""
    def __init__(self):
        self.user_input = ""
        self.user_id = ""
        self.retrieved_context = []
        self.use_remote = False
        self.final_response = ""
        self.memory_items = []


# Global model loading (lazy initialization)
_local_tokenizer = None
_local_model = None


def get_local_model():
    """Lazy load local model"""
    global _local_tokenizer, _local_model
    if _local_tokenizer is None or _local_model is None:
        print("Loading local model...")
        _local_tokenizer, _local_model = load_local_model()
    return _local_tokenizer, _local_model


def retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve relevant context from vector store"""
    try:
        results = query_context(state.user_input, top_k=5)
        
        # Format retrieved context for prompt
        context_texts = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadatas = results.get("metadatas", [{}])
                metadata = metadatas[0][i] if metadatas else {}
                source = metadata.get("source", f"memory_{i}")
                context_texts.append(f"[{source}]: {doc}")
        
        state.retrieved_context = context_texts
        print(f"Retrieved {len(context_texts)} context items")
        return state
        
    except Exception as e:
        print(f"Context retrieval error: {e}")
        state.retrieved_context = []
        return state


def decide_model_node(state: AgentState) -> AgentState:
    """Decide whether to use local or remote model"""
    context_length = sum(len(ctx.split()) for ctx in state.retrieved_context)
    state.use_remote = should_use_remote(state.user_input, context_length)
    
    model_type = "remote (Qwen2.5-7B)" if state.use_remote else "local"
    print(f"Using {model_type} model")
    return state


def generate_response_node(state: AgentState) -> AgentState:
    """Generate response using appropriate model"""
    try:
        # Build prompt with context
        if state.retrieved_context:
            context_str = "\n".join(state.retrieved_context)
        else:
            context_str = "No relevant context found."
        
        if state.use_remote:
            # Use remote Qwen3-30B for complex tasks
            system_content = ("You are a helpful AI assistant. Use the "
                              "provided context to answer accurately. If "
                              "context is not relevant, answer based on "
                              "your knowledge.")
            user_content = (f"Context:\n{context_str}\n\n"
                           f"Question: {state.user_input}")
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            response = qwen3_infer(messages, max_tokens=512)
            
        else:
            # Use local model for quick responses
            tokenizer, model = get_local_model()
            
            # For DialoGPT, create a more informative conversation format
            if context_str and "No relevant context found" not in context_str:
                # Build a conversation with relevant context
                context_summary = context_str[:300] + "..." if len(context_str) > 300 else context_str
                prompt = f"Based on this information: {context_summary}\n\nUser: {state.user_input}\nAssistant:"
            else:
                # Simple conversation format with helpful starter
                prompt = f"User: {state.user_input}\nAssistant:"
            
            response = generate_local(
                tokenizer, model, prompt, max_new_tokens=128
            )
            
            # Post-process response to make it more helpful
            if response and len(response.strip()) > 0:
                # If response is too short or unhelpful, provide context
                if len(response.strip()) < 10 or response.strip().lower() in ["yes", "no", "ok", "sure"]:
                    response = f"I understand your question about '{state.user_input}'. {response} Could you provide more details so I can give you a better answer?"
            else:
                response = "I'd be happy to help you with that. Could you provide a bit more context or rephrase your question?"
        
        state.final_response = response
        return state
        
    except Exception as e:
        print(f"Response generation error: {e}")
        state.final_response = f"Sorry, I encountered an error: {str(e)}"
        return state


def save_memory_node(state: AgentState) -> AgentState:
    """Save important information to memory"""
    try:
        # Check if user wants to save something specific
        user_text = state.user_input.lower()
        
        # Rule: save lines starting with "remember:"
        if "remember:" in user_text:
            lines = state.user_input.split('\n')
            for line in lines:
                if line.lower().strip().startswith("remember:"):
                    # Remove "remember:" prefix
                    memory_text = line[9:].strip()
                    metadata = {
                        "source": "user_request",
                        "timestamp": time.time(),
                        "user_id": state.user_id
                    }
                    add_context(memory_text, metadata)
                    state.memory_items.append(memory_text)
                    print(f"Saved to memory: {memory_text}")
        
        # Optional: save conversation for future context
        conversation_text = (f"User asked: {state.user_input}\n"
                             f"Assistant replied: {state.final_response}")
        metadata = {
            "source": "conversation",
            "timestamp": time.time(),
            "user_id": state.user_id,
            "type": "qa_pair"
        }
        add_context(conversation_text, metadata)
        
        return state
        
    except Exception as e:
        print(f"Memory save error: {e}")
        return state


def run_agent(
    user_input: str,
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Main agent pipeline: retrieve -> decide -> generate -> save
    
    Args:
        user_input: User's question/request
        user_id: User identifier for memory separation
    
    Returns:
        Dictionary with response and metadata
    """
    # Initialize state
    state = AgentState()
    state.user_input = user_input
    state.user_id = user_id
    
    # Run pipeline
    start_time = time.time()
    
    # Step 1: Retrieve context
    state = retrieve_context_node(state)
    
    # Step 2: Decide model
    state = decide_model_node(state)
    
    # Step 3: Generate response
    state = generate_response_node(state)
    
    # Step 4: Save memory
    state = save_memory_node(state)
    
    end_time = time.time()
    
    return {
        "response": state.final_response,
        "model_used": "remote" if state.use_remote else "local",
        "context_items": len(state.retrieved_context),
        "memory_saved": state.memory_items,
        "processing_time": round(end_time - start_time, 2)
    }


# Testing function
if __name__ == "__main__":
    # Test the agent
    test_result = run_agent("Hello, what can you help me with?", "test_user")
    print("Agent result:", test_result)
