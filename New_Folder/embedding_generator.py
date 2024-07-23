from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings(text_list, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embedding_model.embed_documents(text_list)
    return embeddings
