#!/usr/bin/env python3
"""
Build FAISS index using sentence-transformers (no OpenAI required).
Save index to /content/faiss_index/multimodal_text_index.faiss
and meta to /content/faiss_index/multimodal_meta.json
"""
import json, argparse
from pathlib import Path
import numpy as np
import faiss

def main(chunks_path):
    CHUNKS_PATHS = [
        chunks_path,
        "/content/drive/MyDrive/multimodal_rag/chunks/sample_1_chunks.jsonl",
        "/content/sample_chunks.jsonl",
        "/content/sample_multimodal_chunks.jsonl"
    ]
    CHUNKS_PATH = next((p for p in CHUNKS_PATHS if Path(p).exists()), None)
    if not CHUNKS_PATH:
        raise FileNotFoundError("No chunks JSONL found in known locations. Place your chunks file at one of the candidate paths.")

    print("Loading chunks from:", CHUNKS_PATH)
    chunks_all = [json.loads(line) for line in open(CHUNKS_PATH, "r", encoding="utf8")]

    EMBED_MODALITIES = {"text", "table", "caption", "ocr", "paragraph", "table_row"}
    items = [c for c in chunks_all if (c.get("modality") in EMBED_MODALITIES) and c.get("text", "").strip() != ""]
    if len(items) == 0:
        items = [c for c in chunks_all if c.get("text", "").strip() != ""]
        print(f"Fallback: Selected all chunks with non-empty text (len = {len(items)})")
    else:
        print(f"Selected {len(items)} chunks with modalities in {EMBED_MODALITIES}")

    if len(items) == 0:
        raise RuntimeError("No chunks with text found to embed. Check your chunks file.")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "Could not import sentence-transformers. In Colab run the pip install cell. Original error: " + str(e)
        )

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_texts(texts):
        embs = st_model.encode(texts, convert_to_numpy=True)
        return embs.astype(np.float32)

    texts = [c.get("text","") for c in items]
    meta = []
    for i,c in enumerate(items):
        meta.append({
            "idx": i,
            "doc_id": c.get("doc_id"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "modality": c.get("modality"),
            "image_path": c.get("image_path"),
            "bbox": c.get("bbox"),
            "text": c.get("text","")[:2000]
        })

    print("Computing embeddings for", len(texts), "items...")
    embs = embed_texts(texts)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    d = embs.shape[1]
    print("Embedding dim:", d)

    OUT_DIR = Path("/content/faiss_index")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH = OUT_DIR / "multimodal_text_index.faiss"
    META_PATH = OUT_DIR / "multimodal_meta.json"

    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    print("Saved FAISS index to:", INDEX_PATH)

    with open(META_PATH, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved meta to:", META_PATH)
    print("DONE: You can now run the Gradio / retrieval cell (app/ui_gradio.py).")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="/content/drive/MyDrive/multimodal_rag/chunks/sample_1_chunks.jsonl")
    args = ap.parse_args()
    main(args.chunks)
