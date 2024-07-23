from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_and_chunk_pdf(file_path, chunk_size=1000, overlap=200):
    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Combine all text
    text = " ".join(doc.page_content for doc in documents)
    
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    
    return chunks

if __name__ == "__main__":
    file_path = 'path_to_your_pdf.pdf'  # Update with the path to your PDF
    chunks = read_and_chunk_pdf(file_path)
    print(f"Number of chunks: {len(chunks)}")
    print("Chunks:")
    for i, chunk in enumerate(chunks[:5]):  # Display the first 5 chunks for brevity
        print(f"Chunk {i+1}: {chunk}\n")
