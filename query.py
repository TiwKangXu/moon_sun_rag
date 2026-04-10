# query.py
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from rag.pipeline import get_docs

from rag.retriever import retrieve
from rag.reranker import rerank

client = OpenAI()

db = chromadb.PersistentClient(path="db")
collection = db.get_collection("finance")

query = "What drove revenue growth?"

docs = get_docs(collection, query, k=20)

print("\n=== RETRIEVED ===")
for d in docs:
    print("\nFULL:\n", d)

context = "\n\n".join(docs)

prompt = f"""
You are a financial analyst.

STRICT RULES:
- Only use information explicitly found in the context.
- If a number is not present in the context, say "Not found".
- Do NOT infer, estimate, or use prior knowledge.
- For every number, quote the exact sentence from the context.

Format:
- Primary drivers:
- Supporting evidence (with quotes):

Context:
{context}

Question: {query}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n=== ANSWER ===")
print(response.choices[0].message.content)