import json
import os

import numpy as np
import ollama

DATA_DIR = "data"
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "processed_knowledge.json")

MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

_client = ollama.Client(timeout=None)
_knowledge_cache = None


def _load_knowledge():
    global _knowledge_cache
    if _knowledge_cache is None:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            _knowledge_cache = json.load(f)
    return _knowledge_cache


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def search_context(query, top_n=2):
    knowledge = _load_knowledge()

    resp = _client.embeddings(model=EMBED_MODEL, prompt=query)
    q_emb = resp["embedding"]

    scored = []
    for page in knowledge:
        emb = page.get("embedding")
        if emb is None:
            continue
        score = _cosine_similarity(q_emb, emb)
        scored.append((score, page))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [page for _, page in scored[:top_n]]


def _build_messages(user_query):
    pages = search_context(user_query, top_n=4)

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
    response = _client.chat(model=MODEL, messages=_build_messages(user_query))
    return response["message"]["content"]


def generate_answer_stream(user_query):
    stream = _client.chat(
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
