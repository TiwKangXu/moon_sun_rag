# query.py
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from rag.retriever import retrieve
from rag.reranker import rerank

client = OpenAI()

db = chromadb.PersistentClient(path="db")
collection = db.get_collection("finance")

query = "What drove revenue growth?"

docs = retrieve(collection, query)
docs = rerank(query, docs)

print("\n=== RETRIEVED ===")
for d in docs:
    print(d[:200], "\n---")

context = "\n\n".join(docs)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"""
Answer ONLY using the context below.
Be precise with numbers.
If not found, say 'Not found'.

Context:
{context}

Question: {query}
"""}
    ]
)

print("\n=== ANSWER ===")
print(response.choices[0].message.content)