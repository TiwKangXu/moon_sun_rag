# rag/chunker.py
def chunk_elements(elements, company, year):
    chunks = []

    for el in elements:
        text = el.text.strip()
        if not text:
            continue

        chunks.append({
            "text": text,
            "metadata": {
                "company": company,
                "year": year,
                "type": el.category  # Table / NarrativeText
            }
        })

    return chunks