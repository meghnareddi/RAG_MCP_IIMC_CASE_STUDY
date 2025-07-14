from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from config import embedding_function


def load_documents(path):
    document_loader = UnstructuredPDFLoader(path, mode="elements")
    return document_loader.load()

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def create_embeddings_store(documents, persist_directory="db"):
    filtered_documents = filter_complex_metadata(documents)
    vectordb = Chroma.from_documents(
        filtered_documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectordb.persist()  
    #print("Embeddings created and stored successfully.")
    return vectordb

'''
docs = load_documents("data/")
chunks = chunk_documents(docs)
vectordb = create_embeddings_store(chunks, persist_directory="db")
#print(vectordb._collection.get(limit = 3))
'''