from langchain.text_splitter import TextSplitter

def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
    chunks = text_splitter.split(text)
    return chunks

if __name__ == "__main__":
    text = "Your extracted text here"
    chunks = chunk_text(text)
    print(chunks)