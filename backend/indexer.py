import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import fitz
import ollama

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_knowledge.json")

FILES = [
    "geokniga-geologiya-i-geohimiya-nefti-i-gaza.img.pdf",
    "geokniga-geologiya-i-geohimiya-nefti-i-gaza_6.img.pdf",
]

MIN_TEXT_LEN = 10
ZOOM = 2.0
RENDER_WORKERS = 4
EMBED_WORKERS = 8
MAX_PAGES = 50  # set to None for full indexing
EMBED_MODEL = "nomic-embed-text"
PARA_MARK = "\x00"

_ocr_reader = None
_ollama_client = ollama.Client(timeout=None)


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("Инициализирую EasyOCR (русский, GPU/CUDA)...")
        _ocr_reader = easyocr.Reader(["ru"], gpu=True)
    return _ocr_reader


def _render_page(args):
    file_path, page_index = args
    doc = fitz.open(file_path)
    page = doc[page_index]
    raw_text = page.get_text("text")
    image = None
    source_kind = "текст"

    if len(raw_text.strip()) < MIN_TEXT_LEN:
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        ).copy()
        raw_text = None
        source_kind = "OCR"

    doc.close()
    return page_index + 1, raw_text, image, source_kind


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n[ \t]*\n+", PARA_MARK, text)
    text = text.replace("\n", " ")
    text = text.replace(PARA_MARK, "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n\n *", "\n\n", text)
    return text.strip()


MAX_EMBED_CHARS = 4000

def _embed_one(text):
    resp = _ollama_client.embeddings(model=EMBED_MODEL, prompt=text[:MAX_EMBED_CHARS])
    return resp["embedding"]


def embed_batch(texts):
    """Generate embeddings in parallel using ThreadPoolExecutor."""
    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = {executor.submit(_embed_one, t): i for i, t in enumerate(texts)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            embeddings[idx] = future.result()
            done += 1
            if done % 20 == 0 or done == len(texts):
                print(f"  эмбеддинги: {done}/{len(texts)}")
    return embeddings


def process_file(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Файл не найден, пропускаю: {file_path}")
        return []

    doc = fitz.open(file_path)
    total = doc.page_count
    doc.close()
    limit = min(total, MAX_PAGES) if MAX_PAGES else total
    print(f"\n{filename}: {total} страниц, обрабатываю первые {limit} — рендеринг ({RENDER_WORKERS} потоков)...")

    limit = min(total, MAX_PAGES) if MAX_PAGES else total
    args = [(file_path, i) for i in range(limit)]
    rendered = {}
    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as executor:
        for page_num, raw_text, image, source_kind in executor.map(_render_page, args):
            rendered[page_num] = (raw_text, image, source_kind)

    reader = get_ocr_reader()
    results = []
    for page_num in sorted(rendered):
        raw_text, image, source_kind = rendered[page_num]
        if image is not None:
            lines = reader.readtext(image, detail=0, paragraph=True)
            raw_text = "\n".join(lines)
        text = clean_text(raw_text or "")
        results.append({"source": filename, "page": page_num, "text": text})
        print(f"  стр. {page_num:>4}/{total} ({source_kind})")

    return results


def main():
    knowledge = []
    for filename in FILES:
        knowledge.extend(process_file(filename))

    # Generate embeddings in parallel via Ollama (CUDA/RTX, no timeout)
    texts = [p["text"] for p in knowledge]
    print(f"\nГенерация эмбеддингов ({EMBED_MODEL}, {EMBED_WORKERS} потоков, timeout=None)...")
    embeddings = embed_batch(texts)

    for page, emb in zip(knowledge, embeddings):
        page["embedding"] = emb

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Сохранено страниц: {len(knowledge)} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
