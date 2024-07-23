from pdf_processor import read_and_chunk_pdf
from embedding_generator import get_embeddings
from qdrant_manager import QdrantManager

def main(file_path, query_text):
    # Step 1: Read and chunk PDF using LangChain
    chunks = read_and_chunk_pdf(file_path)
    print(f"Read and chunked PDF into {len(chunks)} chunks")
    print("Chunks:")
    for i, chunk in enumerate(chunks[:5]):  # Display the first 5 chunks for brevity
        print(f"Chunk {i+1}: {chunk}\n")
    
    # Step 2: Get embeddings using LangChain and Hugging Face
    embeddings = get_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Step 3: Initialize Qdrant and store embeddings
    qdrant_manager = QdrantManager()
    qdrant_manager.initialize_qdrant(vector_size=len(embeddings[0]))
    qdrant_manager.store_embeddings(embeddings, chunks)
    print("Stored embeddings in Qdrant")
    
    # Step 4: Perform similarity search using LangChain retriever
    retriever = qdrant_manager.get_retriever()
    query_embedding = get_embeddings([query_text])[0]
    search_results = retriever.retrieve(query_embedding)
    
    # Display results
    print("Search results:")
    for result in search_results:
        print(f"Chunk: {result.metadata['chunk']}, Score: {result.score}")

if __name__ == "__main__":
    file_path = 'path_to_your_pdf.pdf'  # Update with the path to your PDF
    query_text = 'Your query text here'  # Update with your query text
    main(file_path, query_text)
