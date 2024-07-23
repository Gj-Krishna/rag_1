from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.vectorstores.base import VectorStoreRetriever

class QdrantManager:
    def __init__(self, collection_name="pdf_chunks"):
        self.client = QdrantClient("http://localhost:6333")  # Change this to your Qdrant URL
        self.collection_name = collection_name

    def initialize_qdrant(self, vector_size):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vector_size=vector_size,
            distance="Cosine"
        )

    def store_embeddings(self, embeddings, chunks):
        points = [{"id": i, "vector": embeddings[i], "payload": {"chunk": chunks[i]}} for i in range(len(embeddings))]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def get_retriever(self):
        vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embedding_dim=len(self.client.collection_info(self.collection_name)["config"]["vector_size"]),
            distance="Cosine"
        )
        return VectorStoreRetriever(vector_store=vector_store)
