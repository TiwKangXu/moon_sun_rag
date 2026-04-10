# rag/parser.py
from unstructured.partition.pdf import partition_pdf

def parse_pdf(path):
    elements = partition_pdf(path)
    return elements