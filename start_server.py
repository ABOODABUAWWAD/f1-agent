#!/usr/bin/env python3
"""
Server startup script for testing
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    try:
        from src.main import app
        import uvicorn
        
        print("🚀 Starting Anigma F1 AI Agent Server...")
        print("📍 Server will be available at: http://localhost:8001")
        print("📖 API documentation at: http://localhost:8001/docs")
        print("❤️  Health check at: http://localhost:8001/health")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()