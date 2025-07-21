import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import re

TEMPLATE_PATH = 'case_study_document_raw_template.txt'
FAISS_INDEX_PATH = 'company_faiss.index'
CHUNK_MAP_PATH = 'company_chunk_map.json'
EMBED_MODEL = 'BAAI/bge-base-en-v1.5'
OUTPUT_PATH = 'company_filled_final.txt'
GROQ_API_KEY = 'GROK_API_KEY'
GROQ_MODEL = 'llama3-70b-8192'
TOP_K = 4

# --- Utility Functions ---
def load_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin1') as f:
            return f.read()

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin1') as f:
            return json.load(f)

def find_bracketed_fields(template):
    # Find all [Field] style placeholders
    return re.findall(r'\[([^\[\]]+)\]', template)

def find_section_c_fields(template):
    # Extract Section C table fields (lines after 'Summary Terms' until next section or end)
    lines = template.splitlines()
    c_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Summary Terms'):
            c_start = i
            break
    if c_start is None:
        return []
    fields = []
    for line in lines[c_start+1:]:
        if not line.strip() or line.strip().startswith('---'):
            continue
        if ':' in line:
            field = line.split(':', 1)[0].strip()
            fields.append(field)
        elif line.strip() and not any(x in line for x in ['(', ')', '[', ']']):
            fields.append(line.strip())
        # Stop if we hit a new section
        if line.strip().startswith('Underwriters') or line.strip().startswith('Gross Spread'):
            break
    return fields

def fill_template(template, blank_to_answer):
    # Replace [Field] placeholders
    for blank, answer in blank_to_answer.items():
        template = template.replace(f'[{blank}]', answer)
    # For Section C fields, replace blank lines after field names
    for field, answer in blank_to_answer.items():
        # Replace lines where the field is followed by a blank line
        template = re.sub(rf'({field}\n)(\s*\n)', rf'\1{answer}\n', template)
    return template

# --- RAG Functions ---
def embed_query(query, model):
    return model.encode([query], normalize_embeddings=True)[0].astype('float32')

def retrieve_chunks(query, model, faiss_index, chunk_map, top_k=TOP_K):
    query_emb = embed_query(query, model)
    D, I = faiss_index.search(np.expand_dims(query_emb, axis=0), top_k)
    return [chunk_map[i] for i in I[0]]

# --- LLM Query ---
def query_llm_groq(prompt, api_key, model=GROQ_MODEL, max_retries=5):
    import time
    client = Groq(api_key=api_key)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise

# --- Main Logic ---
def main():
    template = load_file(TEMPLATE_PATH)
    chunk_map = load_json(CHUNK_MAP_PATH)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    embed_model = SentenceTransformer(EMBED_MODEL)

    bracketed_fields = find_bracketed_fields(template)
    section_c_fields = find_section_c_fields(template)
    all_fields = list(dict.fromkeys(bracketed_fields + section_c_fields))  # preserve order, remove duplicates

    blank_to_answer = {}
    blank_to_sources = {}

    for field in all_fields:
        # Choose prompt style based on field type
        if 'Information about' in field or 'describe' in field.lower() or 'overview' in field.lower():
            # Narrative field
            prompt = (
                f"You are a top finance professional. "
                f"This case study is for an investment banking company. "
                f"Given the following context from official company documents, write a detailed, accurate, and professional paragraph for the following field. Include relevant statistics, financials, and details as appropriate."
                f"\nField: {field}\n"
                f"Context Chunks:\n"
            )
        else:
            # Table/short field
            prompt = (
                f"You are a top finance professional. "
                f"This case study is for an investment banking company. "
                f"Given the following context from official company documents, extract the most accurate and factual value for the field below. Include relevant statistics or numbers if available."
                f"\nField: {field}\n"
                f"Context Chunks:\n"
            )
        # Retrieve relevant chunks
        retrieved = retrieve_chunks(field, embed_model, faiss_index, chunk_map, top_k=TOP_K)
        context = '\n---\n'.join([c['text'] for c in retrieved])
        # Compose LLM prompt
        full_prompt = prompt + context + "\nIf the answer is not in the context, say 'NOT FOUND'."
        answer = query_llm_groq(full_prompt, GROQ_API_KEY)
        blank_to_answer[field] = answer
        blank_to_sources[field] = retrieved

    # Fill the template
    filled = fill_template(template, blank_to_answer)

    # Append reference section (clean, organized)
    filled += '\n\n---\nSOURCE CHUNKS USED FOR EACH FIELD:\n'
    for field, sources in blank_to_sources.items():
        filled += f"\n[{field}]:\n"
        for i, src in enumerate(sources):
            filled += f"  **PDF: {src['pdf']} | Chunk: {src['chunk_id']}**\n"
            filled += f"    {src['text'][:500]}...\n"

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(filled)

if __name__ == "__main__":
    main() 