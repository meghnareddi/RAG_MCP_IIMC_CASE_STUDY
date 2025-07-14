from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from config import embedding_function

def load_documents(path):
    docs = PdfReader(path)
    number_of_pages = len(docs.pages)
    print(f"Total pages: {number_of_pages}")

    content = []
    for i in range(number_of_pages):
        page = docs.pages[i]
        text = page.extract_text()
        if text and text.strip():
            content.append(Document(page_content=text.strip(), metadata={"page_number": i+1}))

    return content


def chunk_documents(documents, chunk_size=500, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def create_embeddings_store(documents, persist_directory="db"):
    vectordb = Chroma.from_documents(
        documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectordb.persist()  
    #print("Embeddings created and stored successfully.")
    return vectordb
