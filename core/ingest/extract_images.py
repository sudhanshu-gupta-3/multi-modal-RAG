# Extract embedded images and render pages, append image chunks to chunks JSONL and update meta
import os, json, io
from pathlib import Path
import fitz   # PyMuPDF
from PIL import Image

def extract_images_and_update(pdf_path, chunks_jsonl_path, out_image_dir="/content/sample_images", page_images_dir="/content/page_images", faiss_meta_path="/content/faiss_index/multimodal_meta.json"):
    PDF_PATH = pdf_path
    CHUNKS_PATH = chunks_jsonl_path
    OUT_IMAGE_DIR = out_image_dir
    PAGE_IMAGES_DIR = page_images_dir
    FAISS_META_PATH = faiss_meta_path

    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PAGE_IMAGES_DIR, exist_ok=True)

    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
    if not Path(CHUNKS_PATH).exists():
        raise FileNotFoundError(f"Chunks JSONL not found at {CHUNKS_PATH}")

    doc = fitz.open(PDF_PATH)
    n_pages = doc.page_count

    embedded_images = []
    for pno in range(n_pages):
        page = doc[pno]
        imglist = page.get_images(full=True)
        if not imglist:
            continue
        for img_index, img in enumerate(imglist):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            image_name = f"{Path(PDF_PATH).stem}_p{pno+1}_embedded_{img_index}.{ext}"
            image_path = str(Path(OUT_IMAGE_DIR) / image_name)
            with open(image_path, "wb") as fh:
                fh.write(image_bytes)
            embedded_images.append({"page": pno+1, "image_path": image_path, "ext": ext})

    zoom = 2.0
    for pno in range(n_pages):
        page = doc[pno]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        out_png = str(Path(PAGE_IMAGES_DIR) / f"page_{pno+1}.png")
        pix.save(out_png)

    new_chunks = []
    base_doc_id = Path(PDF_PATH).stem or "sample_1"
    for i, im in enumerate(embedded_images):
        page = im["page"]
        ipath = im["image_path"]
        image_chunk = {
            "doc_id": base_doc_id,
            "page": page,
            "chunk_id": f"{base_doc_id}_p{page}_embedded_img_{i}",
            "modality": "image",
            "text": "",
            "bbox": None,
            "image_path": ipath
        }
        new_chunks.append(image_chunk)
        ocr_chunk = {
            "doc_id": base_doc_id,
            "page": page,
            "chunk_id": f"{base_doc_id}_p{page}_embedded_img_{i}_ocr",
            "modality": "ocr",
            "text": "",
            "bbox": None,
            "image_path": ipath
        }
        new_chunks.append(ocr_chunk)

    for pno in range(1, n_pages+1):
        ipath = str(Path(PAGE_IMAGES_DIR) / f"page_{pno}.png")
        page_img_chunk = {
            "doc_id": base_doc_id,
            "page": pno,
            "chunk_id": f"{base_doc_id}_p{pno}_page_image",
            "modality": "page_image",
            "text": "",
            "bbox": None,
            "image_path": ipath
        }
        new_chunks.append(page_img_chunk)

    with open(CHUNKS_PATH, "a", encoding="utf8") as fout:
        for c in new_chunks:
            fout.write(json.dumps(c, ensure_ascii=False) + "\\n")

    meta = []
    if Path(FAISS_META_PATH).exists():
        with open(FAISS_META_PATH, "r", encoding="utf8") as f:
            meta = json.load(f)

    start_idx = len(meta)
    for i,c in enumerate(new_chunks):
        meta_entry = {
            "idx": start_idx + i,
            "doc_id": c.get("doc_id"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "modality": c.get("modality"),
            "image_path": c.get("image_path"),
            "bbox": c.get("bbox"),
            "text": c.get("text","")[:2000]
        }
        meta.append(meta_entry)

    with open(FAISS_META_PATH, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "embedded_images": embedded_images,
        "page_images_dir": PAGE_IMAGES_DIR,
        "appended_chunks": len(new_chunks),
        "faiss_meta_count": len(meta)
    }
