# rag/embedder.py
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

def embed(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding