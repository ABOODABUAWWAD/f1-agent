#!/usr/bin/env python3
print("Testing imports...")

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import uvicorn  
    print("✅ Uvicorn imported successfully")
except ImportError as e:
    print(f"❌ Uvicorn import failed: {e}")

try:
    import chromadb
    print("✅ ChromaDB imported successfully")
except ImportError as e:
    print(f"❌ ChromaDB import failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers imported successfully")
except ImportError as e:
    print(f"❌ Sentence Transformers import failed: {e}")

try:
    import transformers
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")

print("Import test complete!")