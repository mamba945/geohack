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


def _build_messages(user_query):
    """Собирает контекст из базы и формирует messages для Ollama."""
    pages = search_context(user_query, top_n=4)

    # Склеиваем найденные страницы с указанием книги и страницы
    context_parts = []
    for page in pages:
        context_parts.append(
            f"[Книга: {page['source']}, страница {page['page']}]\n{page['text']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(ничего не найдено)"

    system_prompt = (
        "Ты — строгий локальный геологический ассистент KAGE. "
        "Ты обязан отвечать ТОЛЬКО на русском языке.\n\n"
        "[КРИТИЧЕСКИЙ ЗАПРЕТ НА КИТАЙСКИЙ И ДРУГИЕ ЯЗЫКИ]\n"
        "1. Твой ответ должен состоять ИСКЛЮЧИТЕЛЬНО из букв кириллицы "
        "(русского алфавита), цифр и стандартных знаков препинания.\n"
        "2. Категорически, абсолютно ЗАПРЕЩЕНО использовать любые китайские "
        "иероглифы (漢字, 中文, любые азиатские символы), английские слова "
        "или латиницу.\n"
        "3. Если в твоей голове формулируется иероглиф или иностранное слово, "
        "ТЫ ОБЯЗАН СТЕРЕТЬ ЕГО и заменить точным русским переводом или аналогом. "
        "За появление хотя бы одного иероглифа система будет аварийно "
        "остановлена.\n\n"
        "[ПРАВИЛО КРАТКОСТИ И СТОП-СИГНАЛ]\n"
        "4. Отвечай строго по существу предоставленного контекста. Как только "
        "ты ответил на вопрос пользователя — СРАЗУ прекращай генерацию.\n"
        "5. Категорически запрещено писать финальные резюме, выводы, подводить "
        "итоги или добавлять пояснения от себя после основного ответа. "
        "Ответил на пункты — остановись.\n\n"
        "[ПРАВИЛО ОТСУТСТВИЯ ИНФОРМАЦИИ]\n"
        "6. Если в предоставленном контексте (4 страницах) нет прямого ответа "
        "на вопрос, ты обязан выдать ровно одну строку: 'В базе знаний нет "
        "точной информации по этому вопросу.' и больше ничего не генерировать."
    )

    user_prompt = (
        "Текст из геологических книг (4 страницы):\n"
        f"{context}\n\n"
        f"Вопрос: {user_query}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(user_query):
    """Собирает контекст и просит локальную модель ответить по нему."""
    response = ollama.chat(model=MODEL, messages=_build_messages(user_query))
    return response["message"]["content"]


def generate_answer_stream(user_query):
    """Потоковая версия: отдаёт ответ модели по кускам (для st.write_stream).

    Используется stream=True, поэтому первые слова появляются на экране почти
    сразу, не дожидаясь полной генерации (важно для CPU, где ответ идёт долго).
    """
    stream = ollama.chat(
        model=MODEL,
        messages=_build_messages(user_query),
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


if __name__ == "__main__":
    question = "Кто авторы учебника 1982 года?"
    print(f"Вопрос: {question}\n")
    print(generate_answer(question))
