from ollama import Ollama

def generate_embeddings(text_chunks, api_key):
    ollama = Ollama(api_key=api_key)
    embeddings = []
    for chunk in text_chunks:
        embedding = ollama.embed(text=chunk)
        embeddings.append(embedding)
    return embeddings

if __name__ == "__main__":
    text_chunks = ["chunk1", "chunk2"]  # Replace with actual chunks
    api_key = "your_ollama_api_key"
    embeddings = generate_embeddings(text_chunks, api_key)
    print(embeddings)
