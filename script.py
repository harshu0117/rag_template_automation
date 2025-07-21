import os
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

# Parameters for chunking
CHUNK_SIZE = 800  # words
CHUNK_OVERLAP = 160  # words (20%)
PDF_DIR = 'pdf'
CHUNK_OUTPUT = 'blbd_pdf_chunks.json'


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception as e:
            print(f"Error extracting page: {e}")
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks


def process_all_pdfs(pdf_dir: str) -> Dict[str, List[str]]:
    pdf_chunks = {}
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, fname)
            print(f"Processing {fname}...")
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            pdf_chunks[fname] = chunks
            print(f"  Extracted {len(chunks)} chunks.")
    return pdf_chunks


def save_chunks(chunks: Dict[str, List[str]], out_path: str):
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    pdf_chunks = process_all_pdfs(PDF_DIR)
    save_chunks(pdf_chunks, CHUNK_OUTPUT)
    print(f"All chunks saved to {CHUNK_OUTPUT}")


if __name__ == "__main__":
    main()
