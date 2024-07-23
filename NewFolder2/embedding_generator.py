from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings(text_list, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embedding_model.embed_documents(text_list)
    return embeddings

if __name__ == "__main__":
    text_list = ["This is a sample text.", "Here is another piece of text."]
    embeddings = get_embeddings(text_list)
    print(f"Number of embeddings: {len(embeddings)}")
    print("First embedding:", embeddings[0] if embeddings else "No embeddings found")
