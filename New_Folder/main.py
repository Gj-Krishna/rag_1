from pdf_processor import read_pdf, chunk_text
from embedding_generator import get_embeddings
from qdrant_manager import QdrantManager

def main(file_path, query_text):
    # Step 1: Read and chunk PDF
    text = read_pdf(file_path)
    chunks = chunk_text(text)
    
    # Step 2: Get embeddings using LangChain and Hugging Face
    embeddings = get_embeddings(chunks)
    
    # Step 3: Initialize Qdrant and store embeddings
    qdrant_manager = QdrantManager()
    qdrant_manager.initialize_qdrant(vector_size=len(embeddings[0]))
    qdrant_manager.store_embeddings(embeddings, chunks)
    
    # Step 4: Perform similarity search using LangChain retriever
    retriever = qdrant_manager.get_retriever()
    query_embedding = get_embeddings([query_text])[0]
    search_results = retriever.retrieve(query_embedding)
    
    # Display results
    for result in search_results:
        print(f"Chunk: {result.metadata['chunk']}, Score: {result.score}")

if __name__ == "__main__":
    file_path = 'path_to_your_pdf.pdf'
    query_text = 'Your query text here'
    main(file_path, query_text)
