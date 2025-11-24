#!/usr/bin/env python3
"""
Build CLIP image FAISS index from images referenced in meta.
Saves to /content/faiss_image_index/image_index.faiss and image_meta.json
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np
import faiss

FAISS_META_PATH = "/content/faiss_index/multimodal_meta.json"
OUT_DIR = Path("/content/faiss_image_index")
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUT_DIR / "image_index.faiss"
META_PATH = OUT_DIR / "image_meta.json"

if not Path(FAISS_META_PATH).exists():
    raise FileNotFoundError("Expected FAISS meta at /content/faiss_index/multimodal_meta.json")

meta = json.load(open(FAISS_META_PATH, "r", encoding="utf8"))
image_to_chunks = {}
for m in meta:
    p = m.get("image_path", None)
    if p:
        image_to_chunks.setdefault(p, []).append(m["chunk_id"])

if len(image_to_chunks) == 0:
    raise RuntimeError("No image files found in meta (check that image_path fields exist)")

from sentence_transformers import SentenceTransformer
img_model = SentenceTransformer("clip-ViT-B-32")

image_paths = []
image_pil_list = []
for img_path in sorted(image_to_chunks.keys()):
    if not Path(img_path).exists():
        print("Warning: image file not found:", img_path)
        continue
    image_paths.append(img_path)
    im = Image.open(img_path).convert("RGB")
    image_pil_list.append(im)

if len(image_pil_list) == 0:
    raise RuntimeError("No valid image files found to embed (check file paths).")

print("Embedding", len(image_pil_list), "images...")
img_embs = img_model.encode(image_pil_list, convert_to_numpy=True, show_progress_bar=True)
img_embs = img_embs.astype(np.float32)
norms = np.linalg.norm(img_embs, axis=1, keepdims=True)
norms[norms==0] = 1.0
img_embs = img_embs / norms

d = img_embs.shape[1]
img_index = faiss.IndexFlatIP(d)
img_index.add(img_embs)
faiss.write_index(img_index, str(INDEX_PATH))
print("Saved image FAISS index to", INDEX_PATH)

image_meta = []
for i, path in enumerate(image_paths):
    image_meta.append({
        "idx": i,
        "image_path": path,
        "chunk_ids": image_to_chunks.get(path, [])
    })

with open(META_PATH, "w", encoding="utf8") as f:
    json.dump(image_meta, f, ensure_ascii=False, indent=2)
print("Saved image meta to", META_PATH)
