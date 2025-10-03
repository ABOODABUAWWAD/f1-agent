# Copilot Instructions for Anigma F1 AI Agent

## Project Overview

This is a hybrid local/remote AI conversational agent with persistent memory using RAG (Retrieval Augmented Generation). The system combines fast local models for quick responses with powerful remote models for complex queries, backed by ChromaDB vector storage for conversation memory.

## Architecture & Key Components

```
src/
├── agent/              # Core agent logic and model management
│   ├── agent.py       # Main agent workflow (retrieve → decide → generate → save)
│   ├── model_loader.py # Local model loading (DialoGPT-medium)
│   └── remote_qwen_tool.py # Remote Qwen2.5-7B integration via HF API
├── infra/             # Infrastructure layer
│   └── vector_store.py # ChromaDB + sentence-transformers embeddings
├── presentation/      # User interfaces
│   └── cli_client.py  # Interactive CLI client
└── main.py           # FastAPI server with REST endpoints
```

## Critical Architecture Patterns

### Agent Workflow (LangGraph-style)

The agent follows a node-based workflow:

1. **retrieve_context_node**: Query ChromaDB for relevant context
2. **decide_model_node**: Choose local vs remote based on query complexity
3. **generate_response_node**: Generate using selected model + context
4. **save_memory_node**: Store important information for future use

```python
# Key pattern: Agent state management
class AgentState:
    def __init__(self):
        self.user_input = ""
        self.retrieved_context = []
        self.use_remote = False
        self.final_response = ""
```

### Model Selection Logic

- **Local (DialoGPT)**: Quick responses, conversational style
- **Remote (Qwen2.5-7B)**: Complex analysis, instruction-following
- **Decision factors**: Query complexity, context length, explicit override (`use_remote:true`)

### Memory System (RAG)

- **Storage**: ChromaDB with sentence-transformers embeddings
- **Context Integration**: Retrieved context injected into prompts
- **Automatic Memory**: Conversation history and explicit `remember:` commands

## Development Workflows

### Local Development Setup

```bash
# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Start server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
uv run python tests/comprehensive_test.py
```

### Model Configuration

```python
# src/agent/model_loader.py
LOCAL_MODEL_ID = "microsoft/DialoGPT-medium"  # Current: conversational
# For Q&A, consider: "mistralai/Mistral-7B-Instruct-v0.1"

# src/agent/remote_qwen_tool.py
REMOTE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Requires HF_TOKEN
```

### Environment Variables

```bash
HF_TOKEN=your_huggingface_token  # Required for remote model access
```

## Project-Specific Conventions

### Response Generation Patterns

```python
# DialoGPT expects conversation format with EOS tokens
conversation_text = prompt + tokenizer.eos_token
# Decode only new tokens, not the full conversation
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
```

### Memory Storage Patterns

```python
# Explicit memory: "remember: I like pizza"
# Automatic memory: Store Q&A pairs after responses
# Context retrieval: Always query before generating
```

### API Endpoint Patterns

- `/ask` - Main chat endpoint (includes context retrieval + model selection)
- `/memory/add` - Manual memory storage
- `/memory/search` - Context search
- `/health` - Model availability check

## Common Issues & Solutions

### DialoGPT Response Quality

- **Issue**: Conversational responses, not direct answers
- **Solution**: Switch to instruction-tuned model or improve prompting
- **Current workaround**: Use `use_remote:true` prefix for factual queries

### Import Path Issues

```python
# Pattern for test files and modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

### Vector Store Initialization

```python
# Lazy loading pattern to avoid startup failures
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
```

## Testing Patterns

### Comprehensive Testing

- **Unit Tests**: `tests/test_imports.py`, `tests/test_memory.py`
- **Integration Tests**: `tests/final_test.py`, `tests/comprehensive_test.py`
- **API Testing**: Use curl with JSON files for endpoint testing

### Key Test Scenarios

1. Model loading and generation
2. Vector store operations (add/query context)
3. Agent workflow execution
4. API endpoint functionality
5. CLI client interaction

## Performance Characteristics

### Local Model (DialoGPT-medium)

- **Load Time**: ~10s first load, cached afterwards
- **Response Time**: 0.5-1.0s per response
- **Memory Usage**: ~2GB VRAM (4-bit quantization)
- **Best For**: Quick conversational responses

### Vector Store (ChromaDB)

- **Storage**: Persistent SQLite backend
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Query Time**: <100ms for small datasets
- **Best For**: Context retrieval, conversation memory

## Future Improvements

1. **Better Local Model**: Switch to instruction-tuned model for Q&A
2. **Enhanced Memory**: Implement conversation summarization
3. **Context Ranking**: Improve relevance scoring for retrieved context
4. **Model Caching**: Optimize load times and memory usage
5. **Error Recovery**: Better fallbacks for model failures
