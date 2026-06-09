"""
Извлечение текста из книг в папке data/ через PyMuPDF (fitz).

Логика по каждой странице:
  1. Пробуем достать цифровой текстовый слой через page.get_text("text").
  2. Если слой пустой (книга — отсканированные картинки, ...img.pdf),
     рендерим страницу в изображение и распознаём её EasyOCR.
  3. Любой текст (цифровой или OCR) чистим через clean_text():
     склеиваем разорванные переносами слова, убираем лишние пробелы,
     сохраняем структуру абзацев.

Работаем на CPU. EasyOCR грузится лениво — только если реально
встретился скан, чтобы не тормозить обработку нормальных PDF.

Результат пишется одним файлом в data/processed_knowledge.json.

Зависимости: PyMuPDF, easyocr, numpy.
"""

import json
import os
import re

import numpy as np
import fitz  # PyMuPDF

# Папка с книгами и итоговый файл
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_knowledge.json")

# Книги для обработки
FILES = [
    "geokniga-geologiya-i-geohimiya-nefti-i-gaza.img.pdf",
    "geokniga-geologiya-i-geohimiya-nefti-i-gaza_6.img.pdf",
]

# Сколько первых страниц каждой книги обрабатывать (быстрая сборка базы)
MAX_PAGES = 10

# Ниже этой длины считаем текстовый слой пустым и идём в OCR
MIN_TEXT_LEN = 10

# Увеличение при рендере страницы (2.0 ~ 144 DPI) — выше качество OCR
ZOOM = 2.0

# Плейсхолдер для временной пометки границ абзацев
PARA_MARK = "\x00"

# EasyOCR-ридер инициализируется лениво (при первом скане)
_ocr_reader = None


def get_ocr_reader():
    """Ленивая инициализация EasyOCR (русский язык, CPU)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        print("Инициализирую EasyOCR (русский, CPU)...")
        _ocr_reader = easyocr.Reader(["ru"], gpu=False)
    return _ocr_reader


def ocr_page(page):
    """Рендерит страницу PDF в картинку (numpy) и распознаёт текст."""
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
    # Пиксели в numpy-массив (height, width, каналы) для EasyOCR
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    lines = get_ocr_reader().readtext(image, detail=0, paragraph=True)
    return "\n".join(lines)


def clean_text(text):
    """Чистит сырой текст страницы, сохраняя структуру абзацев."""
    # Нормализуем переносы строк
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Склеиваем слова, разорванные переносом с дефисом в конце строки
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Двойной перенос = граница абзаца, временно помечаем плейсхолдером
    text = re.sub(r"\n[ \t]*\n+", PARA_MARK, text)

    # Одиночные переносы внутри абзаца заменяем на пробел (слова не рвутся)
    text = text.replace("\n", " ")

    # Возвращаем границы абзацев
    text = text.replace(PARA_MARK, "\n\n")

    # Убираем лишние пробелы внутри строк
    text = re.sub(r"[ \t]+", " ", text)

    # Подчищаем пробелы вокруг переносов абзацев
    text = re.sub(r" *\n\n *", "\n\n", text)

    return text.strip()


def main():
    knowledge = []

    for filename in FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Файл не найден, пропускаю: {file_path}")
            continue

        doc = fitz.open(file_path)
        for page_index, page in enumerate(doc):
            page_num = page_index + 1  # человеческая нумерация страниц

            # Ограничение на первые MAX_PAGES страниц каждой книги
            if page_num > MAX_PAGES:
                break

            # 1. Пробуем цифровой текстовый слой
            raw_text = page.get_text("text")
            source_kind = "текст"

            # 2. Пусто (скан) — распознаём через OCR
            if len(raw_text.strip()) < MIN_TEXT_LEN:
                raw_text = ocr_page(page)
                source_kind = "OCR"

            text = clean_text(raw_text)

            knowledge.append(
                {"source": filename, "page": page_num, "text": text}
            )
            print(f"Файл: {filename} | Страница {page_num} обработана ({source_kind})")

        doc.close()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"Готово. Сохранено страниц: {len(knowledge)} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
