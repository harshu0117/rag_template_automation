import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

CHUNKS_PATH = 'company_pdf_chunks.json'
FAISS_INDEX_PATH = 'company_faiss.index'
CHUNK_MAP_PATH = 'company_chunk_map.json'
EMBED_MODEL = 'BAAI/bge-base-en-v1.5'


def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_chunks(pdf_chunks):
    flat_chunks = []
    chunk_map = []
    for fname, chunks in pdf_chunks.items():
        for i, chunk in enumerate(chunks):
            flat_chunks.append(chunk)
            chunk_map.append({'pdf': fname, 'chunk_id': i, 'text': chunk})
    return flat_chunks, chunk_map


def embed_chunks(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    return np.array(embeddings, dtype='float32')


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, path):
    faiss.write_index(index, path)


def save_chunk_map(chunk_map, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)


def main():
    pdf_chunks = load_chunks(CHUNKS_PATH)
    flat_chunks, chunk_map = flatten_chunks(pdf_chunks)
    print(f"Total chunks: {len(flat_chunks)}")
    embeddings = embed_chunks(flat_chunks, EMBED_MODEL)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, FAISS_INDEX_PATH)
    save_chunk_map(chunk_map, CHUNK_MAP_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Chunk map saved to {CHUNK_MAP_PATH}")


if __name__ == "__main__":
    main() 