# rag/pipeline.py

from rag.retriever import retrieve
from rag.reranker import rerank

def get_docs(collection, query, k=20):
    docs = retrieve(collection, query, k=k)
    docs = rerank(query, docs)
    docs = [d for d in docs if len(d) > 150]  # SAME FILTER
    return docs