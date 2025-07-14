from langchain_community.vectorstores import Chroma
from ingestion import *

def retrieved_docs(query):
    vectordb = Chroma(persist_directory="db", embedding_function=embedding_function)
    results = vectordb.similarity_search_with_score(query, k=5)
    for doc, score in results:
        print(f"Similarity Score: {score:.4f}")
        print(doc.page_content[:600])
        print("-" * 50)


retrieved_docs("What are the four major trends McKinsey identifies that will shape the consumer goods industry in the coming years?")