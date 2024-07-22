from ollama import Ollama
from qdrant_client import QdrantClient

def query_embeddings(query_text, ollama_api_key, qdrant_api_key, collection_name='my_collection'):
    ollama = Ollama(api_key=ollama_api_key)
    client = QdrantClient(api_key=qdrant_api_key)
    query_embedding = ollama.embed(text=query_text)
    results = client.search(collection_name, vector=query_embedding, top=5)
    return results

if __name__ == "__main__":
    query_text = 'your query text'
    ollama_api_key = "your_ollama_api_key"
    qdrant_api_key = "your_qdrant_api_key"
    results = query_embeddings(query_text, ollama_api_key, qdrant_api_key)
    print(results)
