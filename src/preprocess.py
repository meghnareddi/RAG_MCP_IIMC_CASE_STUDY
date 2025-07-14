import ingestion
from config import embedding_function

docs = ingestion.load_documents("data/Trends that will Shape the Consumer Goods Industry.pdf")
chunks = ingestion.chunk_documents(docs)
vectordb = ingestion.create_embeddings_store(chunks, persist_directory="db")
print("Vector store loaded successfully.")


'''
print(f"Total chunks: {len(chunks)}")
print(type(chunks))
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i} content:\n", chunk.page_content[:300])
'''