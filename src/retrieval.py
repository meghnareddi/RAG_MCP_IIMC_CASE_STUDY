from langchain_community.vectorstores import Chroma
import ingestion
from config import embedding_function



def retrieved_docs(query):
    vectordb = Chroma(persist_directory="db", embedding_function=embedding_function)
    retrieved_docs = vectordb.similarity_search_with_relevance_scores(query, k=3)
    for doc, score in retrieved_docs:
        print(f"Relevance Score: {score:.4f}")
        print(doc.page_content[:300])
        print("-" * 50)


retrieved_docs("What are the four major trends McKinsey identifies that will shape the consumer goods industry in the coming years?")