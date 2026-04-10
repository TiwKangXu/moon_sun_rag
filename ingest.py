# ingest.py
import os
import chromadb
from chromadb.config import Settings

from rag.parser import parse_pdf
from rag.chunker import chunk_elements
from rag.embedder import embed

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("finance")

DATA_DIR = "data"
doc_id = 0

for file in os.listdir(DATA_DIR):
    if not file.endswith(".pdf"):
        continue

    path = os.path.join(DATA_DIR, file)

    company = file.split("_")[0]
    year = 2024

    print(f"📄 Processing {file}")

    elements = parse_pdf(path)
    chunks = chunk_elements(elements, company, year)

    for c in chunks:
        collection.add(
            documents=[c["text"]],
            metadatas=[c["metadata"]],
            ids=[str(doc_id)],
            embeddings=[embed(c["text"])]
        )
        doc_id += 1

    print(f"✅ Ingested {file}")

print("TOTAL:", collection.count())