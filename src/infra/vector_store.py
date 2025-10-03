import uuid
import hashlib
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("agent_memory")

# Lazy load embedder to avoid startup issues
_embedder = None


def get_embedder():
    """Lazy load sentence transformer model"""
    global _embedder
    if _embedder is None:
        print("Loading sentence transformer model...")
        try:
            # Try the model without specifying the full path first
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load all-MiniLM-L6-v2: {e}")
            try:
                # Try alternative model that might not need auth
                print("Trying alternative model: paraphrase-MiniLM-L6-v2")
                _embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except Exception as e2:
                print(f"Warning: Could not load paraphrase-MiniLM-L6-v2: {e2}")
                print("Using a simple mock embedder for testing...")
                # Create a mock embedder for testing
                _embedder = MockEmbedder()
    return _embedder


class MockEmbedder:
    """Simple mock embedder for testing when real models fail"""
    def encode(self, texts):
        """Create simple hash-based embeddings for testing"""
        embeddings = []
        for text in texts:
            # Create a simple hash-based vector
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # Convert to numbers and normalize
            vector = [
                int(hash_hex[i:i+2], 16) / 255.0
                for i in range(0, min(len(hash_hex), 32), 2)
            ]
            # Pad to 384 dimensions (like all-MiniLM-L6-v2)
            while len(vector) < 384:
                extend_len = min(384-len(vector), len(vector))
                vector.extend(vector[:extend_len])
            embeddings.append(vector[:384])

        return np.array(embeddings)


def add_context(text: str, metadata: Optional[dict] = None):
    """Add text and metadata to the vector store"""
    # Check for duplicates by searching for exact text match
    try:
        existing = collection.get(where_document={"$contains": text})
        if existing and existing['documents']:
            # Check if exact text already exists
            for doc in existing['documents']:
                if doc.strip() == text.strip():
                    print(f"Skipping duplicate: {text[:50]}...")
                    return
    except Exception:
        # Continue if duplicate check fails
        pass
    
    embedder = get_embedder()
    embedding = embedder.encode([text]).tolist()
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=embedding,
        metadatas=[metadata or {}]
    )


def query_context(query: str, top_k: int = 3):
    """Query the vector store for similar contexts"""
    embedder = get_embedder()
    embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results


def clear_test_data():
    """Clear test data from the vector store"""
    try:
        # Get all items with test metadata
        all_items = collection.get()
        if all_items and all_items['metadatas']:
            test_ids = []
            for i, metadata in enumerate(all_items['metadatas']):
                if metadata and metadata.get('test') == True:
                    test_ids.append(all_items['ids'][i])
            
            if test_ids:
                collection.delete(ids=test_ids)
                print(f"Cleared {len(test_ids)} test items from memory")
    except Exception as e:
        print(f"Warning: Could not clear test data: {e}")


def get_collection_stats():
    """Get statistics about the vector store"""
    try:
        all_items = collection.get()
        count = len(all_items['ids']) if all_items and all_items['ids'] else 0
        return {"total_items": count}
    except Exception as e:
        return {"error": str(e)}
