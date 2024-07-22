from qdrant_client import QdrantClient

def store_embeddings(embeddings, api_key, collection_name='my_collection'):
    client = QdrantClient(api_key=api_key)
    client.create_collection(collection_name, dimension=len(embeddings[0]))
    for idx, embedding in enumerate(embeddings):
        client.upsert(
            collection_name,
            [
                {'id': idx, 'vector': embedding}
            ]
        )

if __name__ == "__main__":
    embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Replace with actual embeddings
    api_key = "your_qdrant_api_key"
    store_embeddings(embeddings, api_key)
