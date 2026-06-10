"""
KAGE — интерфейс геологического ассистента на Streamlit.

Запуск (из корня проекта):
    streamlit run app.py

Чат работает поверх backend.search_engine: поиск по локальной базе знаний
data/processed_knowledge.json + ответ локальной модели Ollama (qwen2.5:7b).
Ответ выводится потоково (stream=True) — буквы бегут по экрану по мере генерации.
"""

import subprocess
import sys

import streamlit as st

from backend.search_engine import MODEL, generate_answer_stream

# --- Настройки страницы ---
st.set_page_config(
    page_title="KAGE | Геологический ассистент",
    page_icon="🪨",
    layout="centered",
)

# --- Заголовок проекта ---
st.title("🪨 KAGE")
st.subheader("Интеллектуальный геологический ассистент")
st.caption(
    "Локальный ИИ-помощник по геологии нефти и газа. "
    "Отвечает строго по базе геологических книг — без выхода в интернет."
)
st.divider()

# --- Боковая панель: информация для жюри ---
with st.sidebar:
    st.header("О системе")
    st.markdown(
        """
        **KAGE работает полностью локально** — без внешних API
        и без передачи данных в интернет.

        - 🧠 **Модель:** `qwen2.5:7b` (Ollama, CPU)
        - 📚 **База знаний:** `data/processed_knowledge.json`
        - 🔎 **Поиск:** полнотекстовый по геологическим книгам
        - 🔒 **Приватность:** все данные остаются на машине
        """
    )
    st.divider()

    st.subheader("Обслуживание базы")
    st.caption(
        "Переиндексация заново извлекает текст из книг в папке `data/` "
        "(цифровой слой + OCR для сканов) и пересобирает базу знаний."
    )
    if st.button("🔄 Переиндексировать базу", use_container_width=True):
        with st.spinner("Идёт переиндексация... это может занять несколько минут"):
            result = subprocess.run(
                [sys.executable, "backend/indexer.py"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        if result.returncode == 0:
            st.success("База знаний переиндексирована.")
        else:
            st.error("Ошибка при переиндексации.")
        # Показываем лог работы индексатора
        log = (result.stdout or "") + (result.stderr or "")
        if log.strip():
            st.code(log.strip(), language="text")

# --- История сообщений (сохраняется между перерисовками страницы) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовываем накопленную историю диалога
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Поле ввода вопроса ---
if user_query := st.chat_input("Задайте вопрос по геологии нефти и газа..."):
    # 1. Сохраняем и показываем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Потоково генерируем и выводим ответ ассистента
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(generate_answer_stream(user_query))
        except Exception as exc:  # noqa: BLE001
            answer = (
                "⚠️ Не удалось получить ответ от модели. "
                "Проверьте, что запущен Ollama и установлена модель "
                f"`{MODEL}`.\n\nДетали: {exc}"
            )
            st.error(answer)

    # 3. Сохраняем ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": answer})
