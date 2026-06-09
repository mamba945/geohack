"""
Извлечение текста из отсканированных книг в папке data/.

Логика по каждой странице:
  1. Пробуем достать текст через pypdf.
  2. Если текста почти нет (слепой скан) — рендерим страницу в картинку
     через PyMuPDF (fitz) и распознаём её EasyOCR.

Результат пишется одним файлом в data/processed_knowledge.json.

Зависимости: pypdf, PyMuPDF, easyocr, numpy.
PyMuPDF работает на Windows без стороннего софта (Poppler не нужен).
"""

import json
import os

import numpy as np
import fitz  # PyMuPDF
from pypdf import PdfReader
import easyocr

# Папка с книгами и итоговый файл
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_knowledge.json")

# Книги для обработки
FILES = [
    "geokniga-geologiya-i-geohimiya-nefti-i-gaza.img.pdf",
    "geokniga-geologiya-i-geohimiya-nefti-i-gaza_6.img.pdf",
]

# Ниже этой длины считаем страницу "слепым" сканом и идём в OCR
MIN_TEXT_LEN = 10

# Увеличение при рендере страницы (2.0 ~ 144 DPI) — выше качество OCR
ZOOM = 2.0

# EasyOCR для русского языка (инициализируем один раз)
reader = easyocr.Reader(["ru"])


def ocr_page(fitz_page):
    """Рендерит страницу PDF в картинку (numpy) и распознаёт текст."""
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
    # Пиксели в numpy-массив (height, width, каналы) для EasyOCR
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    lines = reader.readtext(image, detail=0)
    return "\n".join(lines).strip()


def main():
    knowledge = []

    for filename in FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Файл не найден, пропускаю: {file_path}")
            continue

        pdf = PdfReader(file_path)
        doc = fitz.open(file_path)
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1  # человеческая нумерация страниц

            # Быстрое демо: только первые 5 страниц каждой книги
            if page_num > 5:
                break

            text = (page.extract_text() or "").strip()
            if len(text) < MIN_TEXT_LEN:
                text = ocr_page(doc[page_index])

            knowledge.append(
                {"source": filename, "page": page_num, "text": text}
            )
            print(f"Файл: {filename} | Страница {page_num} обработана")

        doc.close()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"Готово. Сохранено страниц: {len(knowledge)} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
