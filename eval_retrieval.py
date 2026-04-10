# eval_retrieval.py

import chromadb
from rag.retriever import retrieve
from rag.reranker import rerank
from eval_data import EVAL_DATA
from rag.pipeline import get_docs


# load DB
db = chromadb.PersistentClient(path="db")
collection = db.get_collection("finance")


def check_hit(docs, keywords):
    docs_text = " ".join(docs).lower()
    return any(k.lower() in docs_text for k in keywords)


from rag.pipeline import get_docs

def evaluate_recall(k=20):
    results = []

    for item in EVAL_DATA:
        query = item["query"]
        keywords = item["gold_keywords"]

        docs = get_docs(collection, query, k=k)  # ✅ unified

        hit = check_hit(docs, keywords)

        results.append({
            "query": query,
            "hit": int(hit)
        })

        print(f"\nQuery: {query}")
        print(f"Hit: {hit}")

    recall = sum(r["hit"] for r in results) / len(results)

    print("\n=== FINAL RECALL ===")
    print(f"Recall@{k}: {recall:.2f}")

    return results

if __name__ == "__main__":
    evaluate_recall(k=20)