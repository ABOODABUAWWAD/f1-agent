#!/usr/bin/env python3
"""Simple startup script for the FastAPI server"""

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