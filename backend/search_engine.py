"""
Локальный поиск по знаниям + ответ через Ollama (без внешних API).

Логика:
  1. search_context — простой полнотекстовый поиск по словам запроса
     в data/processed_knowledge.json (через .count() в нижнем регистре).
  2. generate_answer — собирает контекст и просит локальную модель
     qwen2.5:7b ответить строго по найденному тексту.

Зависимости: ollama (плюс запущенный локально сервер Ollama с моделью qwen2.5:7b).
"""

import json
import os

import ollama

# Итоговый файл со знаниями (его готовит indexer.py)
DATA_DIR = "data"
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "processed_knowledge.json")

# Локальная модель Ollama
MODEL = "qwen2.5:7b"


def search_context(query, top_n=2):
    """Ищет top_n самых релевантных страниц по словам из query."""
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge = json.load(f)

    # Разбиваем запрос на отдельные слова в нижнем регистре
    words = query.lower().split()

    scored = []
    for page in knowledge:
        text = page["text"].lower()
        # Считаем, сколько раз слова запроса встречаются на странице
        score = sum(text.count(word) for word in words)
        if score > 0:
            scored.append((score, page))

    # Сортируем по убыванию релевантности и берём top_n
    scored.sort(key=lambda item: item[0], reverse=True)
    return [page for _, page in scored[:top_n]]


def generate_answer(user_query):
    """Собирает контекст и просит локальную модель ответить по нему."""
    pages = search_context(user_query)

    # Склеиваем найденные страницы с указанием книги и страницы
    context_parts = []
    for page in pages:
        context_parts.append(
            f"[Книга: {page['source']}, страница {page['page']}]\n{page['text']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(ничего не найдено)"

    prompt = (
        "Ты — локальный ИИ-ассистент хакатона KAGE. "
        "Ответь на вопрос, опираясь ТОЛЬКО на этот текст из геологических книг: "
        f"{context}. "
        "Если ответа нет, скажи, что информации не достаточно. "
        "В конце укажи книгу и страницу.\n\n"
        f"Вопрос: {user_query}"
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    question = "Кто авторы учебника 1982 года?"
    print(f"Вопрос: {question}\n")
    print(generate_answer(question))
