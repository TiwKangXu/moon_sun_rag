# rag/retriever.py
from rag.embedder import embed

def retrieve(collection, query, k=10):
    q_emb = embed(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return results["documents"][0]