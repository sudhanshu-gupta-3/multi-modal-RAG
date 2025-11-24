
# Multi-modal RAG - Document Intelligence

Author: Sudhanshu Gupta

## Summary
Multimodal Retrieval Augmented Generation system for complex PDFs containing
text, tables, and images.

## Demo PDF
Sample PDF used:
/content/drive/MyDrive/multimodal_rag/docs/sample.pdf

## Repo Structure
- app/ : demo apps (Gradio or Streamlit)
- core/ingest/ : PDF parsing, OCR, table extraction
- core/retrieval/ : embeddings, FAISS index, fusion logic
- core/generation/ : HuggingFace or OpenAI generation
- docs/ : technical report and video script
- scripts/ : ingestion helpers

## Quick Start
Install:

