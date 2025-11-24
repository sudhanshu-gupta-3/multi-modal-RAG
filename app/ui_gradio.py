# Multimodal RAG Gradio UI
# (Uses local FAISS indices and meta files)
import os, json, requests, base64, io
from pathlib import Path
from PIL import Image
import numpy as np
import faiss
import gradio as gr

# ==== Paths (use your Drive PDF & indices) ====
PDF_PATH = "/content/drive/MyDrive/multimodal_rag/docs/sample.pdf"   # provenance (developer-specified)
TEXT_INDEX_PATH = "/content/faiss_index/multimodal_text_index.faiss"
TEXT_META_PATH = "/content/faiss_index/multimodal_meta.json"
IMAGE_INDEX_PATH = "/content/faiss_image_index/image_index.faiss"
IMAGE_META_PATH = "/content/faiss_image_index/image_meta.json"

# sanity checks
if not Path(PDF_PATH).exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
if not Path(TEXT_INDEX_PATH).exists() or not Path(TEXT_META_PATH).exists():
    raise FileNotFoundError("Text index/meta not found. Run text FAISS build first.")

image_index_exists = Path(IMAGE_INDEX_PATH).exists() and Path(IMAGE_META_PATH).exists()

# load text index + meta
text_index = faiss.read_index(TEXT_INDEX_PATH)
with open(TEXT_META_PATH, "r", encoding="utf8") as f:
    text_meta = json.load(f)
meta_by_chunk = {m["chunk_id"]: m for m in text_meta}

# load image index + meta if present
if image_index_exists:
    image_index = faiss.read_index(IMAGE_INDEX_PATH)
    with open(IMAGE_META_PATH, "r", encoding="utf8") as f:
        image_meta = json.load(f)   # list of {idx, image_path, chunk_ids}
    image_paths = [m["image_path"] for m in image_meta]
else:
    image_meta = []
    image_paths = []

# load encoders (sentence-transformers)
from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
img_encoder = SentenceTransformer("clip-ViT-B-32")

# helpers
def embed_text_query(q):
    v = text_encoder.encode([q], convert_to_numpy=True)[0].astype(np.float32)
    v = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
    return v

def embed_image_pil(pil_img):
    v = img_encoder.encode([pil_img], convert_to_numpy=True)[0].astype(np.float32)
    v = v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
    return v

def retrieve_text_topk(question, top_k=10):
    qv = embed_text_query(question).reshape(1,-1)
    D, I = text_index.search(qv, top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(text_meta): continue
        m = text_meta[idx].copy()
        m["score_text"] = float(score)
        results.append(m)
    return results

def retrieve_image_topk_by_image(pil_img, top_k=10):
    if not image_index_exists:
        return []
    qv = embed_image_pil(pil_img).reshape(1,-1)
    D, I = image_index.search(qv, top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(image_meta): continue
        im = image_meta[idx].copy()
        im["score_image"] = float(score)
        results.append(im)
    return results

def retrieve_image_topk_by_text(question, top_k=10):
    if not image_index_exists:
        return []
    try:
        qv = img_encoder.encode([question], convert_to_numpy=True)[0].astype(np.float32)
        qv = qv / np.linalg.norm(qv) if np.linalg.norm(qv) != 0 else qv
        D, I = image_index.search(qv.reshape(1,-1), top_k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(image_meta): continue
            im = image_meta[idx].copy()
            im["score_image"] = float(score)
            results.append(im)
        return results
    except Exception:
        return []

def normalize_map(scores_map):
    if not scores_map:
        return {}
    keys = list(scores_map.keys())
    vals = np.array([scores_map[k] for k in keys], dtype=np.float32)
    minv = vals.min()
    vals = vals - minv
    maxv = vals.max() if vals.max() != 0 else 1.0
    vals = vals / maxv
    return {keys[i]: float(vals[i]) for i in range(len(keys))}

def fuse_text_image(text_hits, image_hits_mapped, alpha=0.75):
    text_map = {t["chunk_id"]: t.get("score_text", 0.0) for t in text_hits}
    text_norm = normalize_map(text_map)
    image_norm = normalize_map(image_hits_mapped)
    fused = {}
    for k,v in text_norm.items():
        fused[k] = fused.get(k,0.0) + alpha * v
    for k,v in image_norm.items():
        fused[k] = fused.get(k,0.0) + (1-alpha) * v
    for t in text_hits:
        fused.setdefault(t["chunk_id"], fused.get(t["chunk_id"], 0.0))
    fused_list = []
    for cid, score in fused.items():
        m = meta_by_chunk.get(cid)
        if m:
            out = m.copy()
            out["fused_score"] = float(score)
            out["score_text"] = float(out.get("score_text", 0.0))
            out["score_image"] = float(image_hits_mapped.get(cid, 0.0))
            fused_list.append(out)
    fused_list = sorted(fused_list, key=lambda x: -x["fused_score"])
    return fused_list

def image_hits_to_chunk_map(image_hits):
    m = {}
    for im in image_hits:
        score = im.get("score_image", 0.0)
        for cid in im.get("chunk_ids", []):
            m[cid] = max(m.get(cid, 0.0), score)
    return m

def image_b64_html(image_path, max_w=300):
    try:
        with open(image_path, "rb") as fh:
            b = fh.read()
        b64 = base64.b64encode(b).decode("utf8")
        return f"<img src='data:image/png;base64,{b64}' style='max-width:{max_w}px; height:auto'/>"
    except Exception:
        return ""

ROUTER = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN","")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else None
CANDIDATES = ["google/gemma-2-2b-it","HuggingFaceTB/smol-llama-3-1b-instruct","microsoft/Phi-3-mini-4k-instruct"]

def probe_hf_model():
    if not HF_TOKEN:
        return None
    for m in CANDIDATES:
        try:
            r = requests.post(ROUTER, headers=HEADERS, json={"model":m,"messages":[{"role":"user","content":"hi"}],"max_tokens":6}, timeout=10)
            if r.status_code == 200:
                return m
        except Exception:
            pass
    return None

working_model = probe_hf_model()
USE_OPENAI_FALLBACK = bool(os.environ.get("OPENAI_API_KEY")) and (working_model is None)
print("Working HF model:", working_model, " OpenAI fallback:", USE_OPENAI_FALLBACK, " Image index present:", image_index_exists)

def hf_generate(prompt, model, max_tokens=400):
    payload = {"model": model, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens, "temperature":0.0}
    r = requests.post(ROUTER, headers=HEADERS, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def openai_generate(prompt, max_tokens=400):
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=max_tokens)
    return resp["choices"][0]["message"]["content"]

def multimodal_search_and_answer(question, uploaded_image, alpha, top_k=12):
    text_hits = retrieve_text_topk(question, top_k=top_k) if question and question.strip() else []
    image_hits = []
    if uploaded_image is not None:
        if isinstance(uploaded_image, np.ndarray):
            pil = Image.fromarray(uploaded_image.astype('uint8'), 'RGB')
        elif isinstance(uploaded_image, Image.Image):
            pil = uploaded_image
        else:
            pil = Image.open(io.BytesIO(uploaded_image)).convert("RGB")
        image_hits = retrieve_image_topk_by_image(pil, top_k=top_k)
    else:
        if question and question.strip() and image_index_exists:
            image_hits = retrieve_image_topk_by_text(question, top_k=top_k)

    image_chunk_map = image_hits_to_chunk_map(image_hits) if image_hits else {}
    fused = fuse_text_image(text_hits, image_chunk_map, alpha=alpha)

    if not fused:
        if text_hits:
            fused = text_hits[:top_k]
        elif image_hits:
            fallback = []
            for im in image_hits[:top_k]:
                for cid in im.get("chunk_ids", [])[:1]:
                    m = meta_by_chunk.get(cid)
                    if m:
                        m = m.copy()
                        m["fused_score"] = float(im.get("score_image",0.0))
                        fallback.append(m)
            fused = fallback

    context_parts = []
    for r in fused[:8]:
        txt = r.get("text","")
        citation = f"[CITE:{r['chunk_id']}]"
        if txt.strip():
            context_parts.append(f"{citation}\\n{txt}")
    context = "\\n\\n".join(context_parts)
    prompt = (
        f"You are a document assistant. Use ONLY the context below and the document at file://{PDF_PATH}.\\n"
        "When you assert a factual claim, append the citation token (e.g. [CITE:...]) used.\\n"
        "If you cannot find evidence, say so.\\n\\n"
        f"Context:\\n{context}\\n\\nQuestion: {{} }\\n\\nAnswer concisely, then list the Sources with citation tokens."
    ).format("{question}")

    try:
        if working_model:
            gen = hf_generate(prompt.format(question=question), working_model)
        elif USE_OPENAI_FALLBACK:
            gen = openai_generate(prompt.format(question=question))
        else:
            gen = "No generation backend available. Top snippets:\\n\\n"
            for r in fused[:4]:
                snippet = (r.get("text","") or "")[:400].replace("\\n"," ")
                gen += f"- ({r['chunk_id']}) {snippet}...\\n\\n"
    except Exception as e:
        gen = f"Generation error: {e}"

    sources_html = ""
    dropdown_options = []
    for r in fused[:top_k]:
        cid = r["chunk_id"]
        label = f"{cid} (page {r.get('page')}, {r.get('modality')}, score={r.get('fused_score',0):.3f})"
        dropdown_options.append((cid, label))
        snippet = (r.get("text","") or "")[:400].replace("\\n"," ")
        img_html = ""
        if r.get("image_path") and Path(r["image_path"]).exists():
            with open(r["image_path"], "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf8")
                img_html = f"<img src='data:image/png;base64,{b64}' style='max-width:280px;height:auto'/>"
        sources_html += f"### {label}\\n\\n> {snippet}...\\n\\n{img_html}\\n\\n---\\n\\n"

    upload_preview = ""
    if uploaded_image is not None:
        if isinstance(uploaded_image, np.ndarray):
            pil = Image.fromarray(uploaded_image.astype('uint8'), 'RGB')
        elif isinstance(uploaded_image, Image.Image):
            pil = uploaded_image
        else:
            pil = Image.open(io.BytesIO(uploaded_image)).convert("RGB")
        buf = io.BytesIO(); pil.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode("utf8")
        upload_preview = f"<b>Uploaded image (query example):</b><br/><img src='data:image/png;base64,{b64}' style='max-width:360px;height:auto'/><hr/>"

    pdf_iframe = f"<iframe src='file://{PDF_PATH}' width='800' height='520'></iframe>"
    return gen, upload_preview + pdf_iframe + "<br/><hr/>" + sources_html, dropdown_options, (dropdown_options[0][0] if dropdown_options else None)

def open_chunk(chunk_id):
    m = meta_by_chunk.get(chunk_id)
    if not m:
        return "Chunk not found.", ""
    text = m.get("text","")
    img_html = ""
    if m.get("image_path") and Path(m["image_path"]).exists():
        with open(m["image_path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf8")
            img_html = f"<img src='data:image/png;base64,{b64}' style='max-width:640px;height:auto'/>"
    header = f"**{m['chunk_id']}** (page {m.get('page')}, modality {m.get('modality')})\\n\\n"
    return header + text, img_html

with gr.Blocks() as demo:
    gr.Markdown("## Multimodal RAG — Visual search (upload image) + text query + PDF preview")
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Question (optional for image-only search)", placeholder="Ask about the document...")
            uploaded_img = gr.Image(label="Upload an example image (optional) — search visually by example", type="numpy")
            alpha = gr.Slider(label="Text weight (alpha). 1.0=text-only, 0.0=image-only", minimum=0.0, maximum=1.0, value=0.75, step=0.05)
            submit = gr.Button("Search (text/image fusion)")
        with gr.Column(scale=2):
            gr.Markdown("**Provenance PDF**")
            pdf_place = gr.HTML(f"<iframe src='file://{PDF_PATH}' width='360' height='420'></iframe>")

    with gr.Row():
        answer_out = gr.Textbox(label="Answer / Generated output", lines=6)
    with gr.Row():
        sources_out = gr.HTML(label="PDF preview + Sources")
    with gr.Row():
        dropdown = gr.Dropdown(label="Select retrieved chunk to open (then click Open chunk)", choices=[], value=None)
        open_btn = gr.Button("Open chunk")
    with gr.Row():
        full_chunk_md = gr.Markdown()
        full_chunk_img = gr.HTML()

    def on_submit(q, img, a):
        gen, src_html, options, default_cid = multimodal_search_and_answer(q, img, alpha=a, top_k=12)
        combo = [f"{opt[0]} | {opt[1]}" for opt in options]
        default_display = combo[0] if combo else None
        return gen, src_html, gr.Dropdown.update(choices=combo, value=default_display), default_cid

    submit.click(on_submit, inputs=[question, uploaded_img, alpha], outputs=[answer_out, sources_out, dropdown, dropdown])

    def on_open(selection):
        if not selection:
            return "No chunk selected.", ""
        cid = selection.split("|")[0].strip()
        txt, img_html = open_chunk(cid)
        return txt, img_html

    open_btn.click(on_open, inputs=[dropdown], outputs=[full_chunk_md, full_chunk_img])

demo.launch(share=True)
