# Anigma F1 AI Agent 🚀

A hybrid local + remote AI assistant with persistent memory using RAG (Retrieval Augmented Generation). 

## Features

- **Hybrid Model Strategy**: Fast local 7B model + powerful remote Qwen3-30B for complex tasks
- **Persistent Memory**: Chroma vector database with sentence-transformers embeddings  
- **Smart Routing**: Automatically chooses local vs remote model based on complexity
- **FastAPI Server**: RESTful API with automatic docs
- **CLI Client**: Easy interactive chat interface
- **WSL2 Optimized**: Built for Windows with WSL2 + GPU support

## Architecture

```
User Input → Retrieve Context → Decide Model → Generate Response → Save Memory
              ↓                 ↓              ↓                  ↓
           Chroma DB          Local/Remote    RTX 3050 or      Update Vector
           Embeddings         Decision        HF API           Store
```

## Quick Start

### 1. Prerequisites (WSL2 Ubuntu)

```bash
# Ensure WSL2 with Ubuntu and GPU access
# Install Python 3.12+, git, build tools
sudo apt update && sudo apt install -y python3 python3-venv git build-essential
```

### 2. Install Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install all dependencies
uv add fastapi uvicorn[standard] langgraph chromadb sentence-transformers transformers accelerate bitsandbytes huggingface_hub python-dotenv requests
```

### 3. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Login to Hugging Face (for remote model access)
huggingface-cli login
# OR set token manually in .env:
# echo "HF_TOKEN=hf_your_token_here" >> .env
```

### 4. Run the Server

```bash
# Start FastAPI server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Server will be available at:
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Use the CLI Client

```bash
# Interactive chat mode
python cli_client.py

# Single question
python cli_client.py -q "What is quantum computing?"

# Health check
python cli_client.py --health
```

## Usage Examples

### Basic Chat
```bash
👤 You: Hello, what can you help me with?
🤖 I'm your AI assistant! I can help with questions, remember information, and more.
   └─ Model: local | Time: 0.8s | Context: 0 items
```

### Save to Memory
```bash
👤 You: Remember: My favorite programming language is Python
🤖 I've saved that information to memory.
   └─ Model: local | Time: 0.5s | Context: 0 items
   └─ Saved to memory: My favorite programming language is Python
```

### Complex Query (Uses Remote Model)
```bash
👤 You: Explain quantum computing in detail with technical examples
🤖 [Detailed quantum computing explanation...]
   └─ Model: remote (Qwen3-30B) | Time: 3.2s | Context: 1 items
```

### Force Remote Model
```bash
👤 You: use_remote:true What are the latest AI developments?
```

## API Endpoints

- `POST /ask` - Main chat endpoint
- `GET /health` - Server health and model status  
- `POST /memory/add` - Add information to memory
- `GET /memory/search` - Search memory/context
- `GET /docs` - Interactive API documentation

## Model Strategy

### Local Model (RTX 3050)
- **Model**: 7B quantized (4-bit) - DialoGPT-medium (fallback), Mistral-7B, or Qwen2.5-7B
- **Use Cases**: Quick responses, simple questions, private conversations
- **Benefits**: Fast, private, no API costs

### Remote Model (Hugging Face)
- **Model**: Qwen3-Omni-30B-A3B-Instruct  
- **Use Cases**: Complex analysis, detailed explanations, research tasks
- **Benefits**: High-quality responses, latest knowledge

### Decision Logic
Automatically uses remote model when:
- Text contains "use_remote:true"
- Complex keywords detected (analyze, explain in detail, etc.)
- Query + context length > 200 tokens

## Memory System

- **Vector Store**: Chroma with persistent storage
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Auto-Save**: Conversations saved for context
- **Manual Save**: Lines starting with "Remember:" 
- **Retrieval**: Top-5 relevant chunks for each query

## Configuration

### Environment Variables (.env)
```bash
HF_TOKEN=hf_your_token_here           # Required for remote model
MODEL_CACHE_DIR=./models              # Local model cache
CHROMA_DB_PATH=./chroma_db            # Vector database path
```

### Customization
- Change local model in `src/model_loader.py`
- Adjust decision logic in `src/remote_qwen_tool.py`
- Modify memory rules in `src/agent.py`

## Troubleshooting

### bitsandbytes Installation Issues
```bash
# Ensure CUDA-compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# If still failing, use CPU-only mode (slower)
# Comment out load_in_4bit=True in model_loader.py
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Ensure WSL2 has GPU support
nvidia-smi
```

### Remote Model Errors
- Verify HF_TOKEN is set correctly
- Check Hugging Face API status
- Model may be rate-limited (fallback to local)

## Development

### Project Structure
```
anigma-f1/
├── src/
│   ├── model_loader.py      # Local model management
│   ├── remote_qwen_tool.py  # Remote model API
│   ├── vector_store.py      # Chroma + embeddings
│   └── agent.py             # Main orchestration logic
├── main.py                  # FastAPI server
├── cli_client.py            # CLI interface
├── pyproject.toml           # Dependencies (uv)
└── README.md                # This file
```

### Adding Features
1. **New Memory Rules**: Edit `save_memory_node()` in `agent.py`
2. **Custom Models**: Update `LOCAL_MODEL_ID` in `model_loader.py`
3. **API Endpoints**: Add routes in `main.py`
4. **Decision Logic**: Modify `should_use_remote()` in `remote_qwen_tool.py`

## License

MIT License - feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Ready to chat with your AI agent!** 🤖✨
