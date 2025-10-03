from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

# Import our agent
from src.agent.agent import run_agent
from src.infra.vector_store import add_context, query_context

# Load environment variables
load_dotenv()


# FastAPI app
app = FastAPI(
    title="Anigma F1 AI Agent",
    description="Local + Remote AI Assistant with RAG Memory",
    version="0.1.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    user_id: str
    text: str

class ChatResponse(BaseModel):
    reply: str
    model_used: str
    context_items: int
    processing_time: float
    memory_saved: list = []

class MemoryRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    models_available: Dict[str, bool]

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if HF token is available for remote model
        hf_token_available = bool(os.getenv("HF_TOKEN"))
        
        # For now, assume local model is available (could add actual check)
        local_model_available = True
        
        return HealthResponse(
            status="healthy",
            models_available={
                "local": local_model_available,
                "remote": hf_token_available
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/ask", response_model=ChatResponse)
async def ask_agent(request: ChatRequest):
    """Main chat endpoint - ask the AI agent"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Run the agent
        result = run_agent(request.text, request.user_id)
        
        return ChatResponse(
            reply=result["response"],
            model_used=result["model_used"],
            context_items=result["context_items"],
            processing_time=result["processing_time"],
            memory_saved=result.get("memory_saved", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.post("/memory/add")
async def add_memory(request: MemoryRequest):
    """Add information to memory manually"""
    try:
        add_context(request.text, request.metadata or {})
        return {"status": "success", "message": "Memory added"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory error: {str(e)}")

@app.get("/memory/search")
async def search_memory(query: str, top_k: int = 5):
    """Search memory/context"""
    try:
        results = query_context(query, top_k)
        return {
            "query": query,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Anigma F1 AI Agent API",
        "endpoints": {
            "chat": "/ask",
            "health": "/health", 
            "add_memory": "/memory/add",
            "search_memory": "/memory/search"
        }
    }

# CLI mode for direct testing
if __name__ == "__main__":
    import uvicorn
    print("Starting Anigma F1 AI Agent...")
    print("API will be available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
