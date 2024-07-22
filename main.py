from extract_text import extract_text_from_pdf
from chunk_text import chunk_text
from generate_embeddings import generate_embeddings
from store_embeddings import store_embeddings
from query_embeddings import query_embeddings

def main():
    pdf_path = 'your_document.pdf'
    ollama_api_key = "your_ollama_api_key"
    qdrant_api_key = "your_qdrant_api_key"
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Chunk the text
    chunks = chunk_text(pdf_text)
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, ollama_api_key)
    
    # Store embeddings in Qdrant
    store_embeddings(embeddings, qdrant_api_key)
    
    # Query embeddings
    query_text = 'your query text'
    results = query_embeddings(query_text, ollama_api_key, qdrant_api_key)
    print(results)

if __name__ == "__main__":
    main()
