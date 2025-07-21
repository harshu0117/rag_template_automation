# RAG-Driven Document Automation Pipeline

Automate the extraction, semantic search, and template-driven generation of structured documents from unstructured PDF sources using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs).

---

## 🚀 Overview

This project provides an end-to-end pipeline for generating structured documents (e.g., reports, case studies, summaries) from a collection of PDFs. It leverages state-of-the-art NLP techniques to extract, chunk, embed, and retrieve relevant information, and uses an LLM to fill a customizable template, producing a polished output with clear source references.

---

## 🛠️ Tech Stack

- **Python**: Scripting and automation
- **pypdf**: PDF text extraction
- **sentence-transformers**: Text embedding (e.g., BAAI/bge-base-en-v1.5)
- **FAISS**: Fast vector similarity search
- **Groq LLM API (Llama3-70B)**: Natural language generation (can be swapped for any LLM API)
- **Regular Expressions**: Template parsing and field extraction

---

## 📂 Project Structure

```
.
├── pdf/                          # Input PDFs (source documents)
├── script.py                     # Extracts and chunks text from PDFs
├── embed_chunks.py               # Embeds chunks and builds FAISS index
├── fill_template.py              # Fills the template using RAG + LLM
├── case_study_document_raw_template.txt  # Example template with placeholders
├── blbd_pdf_chunks.json          # Output: Chunked text from PDFs
├── blbd_faiss.index              # Output: FAISS vector index
├── blbd_chunk_map.json           # Output: Mapping of chunks to source
├── blbd_filled_final.txt         # Output: Final filled document
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation (this file)
```

---

## ⚙️ Workflow

1. **Extract & Chunk PDF Text**
   - Run `script.py` to extract text from all PDFs in `pdf/` and split into overlapping chunks.
   - Output: `blbd_pdf_chunks.json`

2. **Embed Chunks & Build Index**
   - Run `embed_chunks.py` to embed all chunks and build a FAISS index for semantic search.
   - Outputs: `blbd_faiss.index`, `blbd_chunk_map.json`

3. **Fill the Template**
   - Run `fill_template.py` to:
     - Identify all fields in the template.
     - Retrieve the most relevant chunks for each field.
     - Query an LLM to generate answers using the retrieved context.
     - Fill the template and append source references.
   - Output: `blbd_filled_final.txt`

---

## 💡 Use Cases

- Automated report generation from research papers, legal documents, or financial filings
- Summarizing and structuring information from large PDF archives
- Creating reference-backed case studies, whitepapers, or compliance documents
- Any workflow requiring extraction of structured data from unstructured PDF sources

---

## 📦 Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your input PDFs in the `pdf/` directory.

---

## 🏃‍♂️ Usage

1. **Extract and chunk PDF text:**
   ```bash
   python script.py
   ```
2. **Embed chunks and build FAISS index:**
   ```bash
   python embed_chunks.py
   ```
3. **Fill the template:**
   ```bash
   python fill_template.py
   ```
4. **Check the output:**  
   The filled document will be saved as `blbd_filled_final.txt`.

---

## 📝 Customization

- **Template:**  
  Edit `case_study_document_raw_template.txt` to change the structure or fields of the output document.
- **Model:**  
  The embedding model and LLM can be changed in the scripts if needed.

---

## 📄 License

MIT License (or your preferred license)

---

## 🙏 Acknowledgements

- [pypdf](https://pypdf.readthedocs.io/)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Groq](https://groq.com/)

---
