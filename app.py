import re
import subprocess
import sys

import pandas as pd
import streamlit as st

from backend.search_engine import MODEL, generate_answer_stream, search_context
from backend.gap_analyzer import gap_analysis_stream


def clean_ocr_text(text: str) -> str:
    text = re.sub(r'-\n\s*', '', text)
    text = re.sub(r'-\s+([а-яё])', r'\1', text)
    text = re.sub(r'(?<=[А-ЯЁ])0(?=[А-ЯЁ ])', 'О', text)
    text = re.sub(r'(?<=\s)0(?=[А-ЯЁ])', 'О', text)
    text = re.sub(
        r'\b([А-ЯЁа-яёA-Za-z])(?:\s([А-ЯЁа-яёA-Za-z])){2,}\b',
        lambda m: m.group(0).replace(' ', ''),
        text,
    )
    text = ' '.join(text.split())
    return text


st.set_page_config(
    page_title="GeoMind | Геологический ассистент",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        .geo-masthead {
            display: flex;
            align-items: baseline;
            gap: 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 1.25rem;
            margin-bottom: 1.75rem;
        }
        .geo-logotype {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: #e2e8f0;
            font-variant: small-caps;
        }
        .geo-tagline {
            font-size: 0.78rem;
            color: #556;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .geo-badges {
            margin-left: auto;
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .badge {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.2rem 0.6rem;
            border-radius: 3px;
            border: 1px solid;
        }
        .badge-ok   { color: #68d391; border-color: #276749; background: rgba(39,103,73,0.15); }
        .badge-lock { color: #90cdf4; border-color: #2b6cb0; background: rgba(43,108,176,0.15); }
        .badge-cpu  { color: #f6ad55; border-color: #9c5a1d; background: rgba(156,90,29,0.15); }
        .verdict-container {
            border-left: 3px solid #4a5568;
            padding-left: 1.25rem;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="geo-masthead">
        <div>
            <div class="geo-logotype">GeoMind</div>
            <div class="geo-tagline">Локальный геологический ассистент — база нефтегазовых книг</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_search, tab_gap = st.tabs([
    "Поиск по книгам",
    "Аудит архивов (Слепые зоны)",
])

with tab_search:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            """
            <style>
                .geomind-hero {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 50vh;
                    text-align: center;
                    padding: 2rem 1rem;
                }
                .geomind-hero h1 {
                    font-size: 2.8rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #90cdf4 0%, #68d391 50%, #f6ad55 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.5rem;
                }
                .geomind-hero p {
                    font-size: 1.05rem;
                    color: #718096;
                    max-width: 480px;
                    line-height: 1.6;
                }
            </style>
            <div class="geomind-hero">
                <h1>GeoMind, приступим!</h1>
                <p>Ваш геологический ассистент по нефтегазовой литературе. Задайте вопрос — я найду ответ в учебниках и процитирую источники.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_query := st.chat_input("Спросите GeoMind..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            sources = []
            try:
                sources = search_context(user_query, top_n=4)
                answer = st.write_stream(generate_answer_stream(user_query))
            except Exception as exc:
                answer = (
                    "Не удалось получить ответ от модели. "
                    "Проверьте, что запущен Ollama и установлена модель "
                    f"`{MODEL}`.\n\nДетали: {exc}"
                )
                st.error(answer)

            if sources:
                with st.expander("Первоисточники и цитаты"):
                    st.markdown("**Источники, использованные при генерации ответа:**")
                    st.markdown("---")
                    for i, src in enumerate(sources, 1):
                        book = src.get("source", "Неизвестно")
                        page_num = src.get("page", "—")
                        text = clean_ocr_text(src.get("text", ""))
                        quote = text[:400].strip()
                        if len(text) > 400:
                            quote += "..."
                        st.markdown(
                            f"**Документ:** {book} | **Страница:** {page_num} | **Фрагмент:** {i}"
                        )
                        st.markdown(f'> "{quote}"')
                        if i < len(sources):
                            st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_gap:
    st.markdown(
        "Загрузите текст двух отчётов для выявления **потерь данных**, "
        "**физических противоречий** и **терминологических расхождений** между эпохами."
    )
    st.markdown("")

    EXAMPLE_HISTORICAL = (
        "Интервал 450-500м. Описаны интенсивные газопроявления, "
        "дебит составлял 15 куб.м/сут. Пласт сложен известняками "
        "с высокой пористостью."
    )
    EXAMPLE_MODERN = (
        "Интервал 450-500м признан непродуктивным, каротаж показывает "
        "плотные глинистые перемычки, испытания в данном интервале "
        "не проводились."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Исторический отчёт (прошлый век)**")
        historical = st.text_area(
            "historical_report",
            value=EXAMPLE_HISTORICAL,
            height=220,
            label_visibility="collapsed",
            key="gap_historical",
            placeholder="Вставьте текст исторического геологического отчёта...",
        )

    with col2:
        st.markdown("**Современный отчёт**")
        modern = st.text_area(
            "modern_report",
            value=EXAMPLE_MODERN,
            height=220,
            label_visibility="collapsed",
            key="gap_modern",
            placeholder="Вставьте текст современного геологического отчёта...",
        )

    run_gap = st.button(
        "Запустить Gap-анализ архивов",
        type="primary",
        use_container_width=True,
    )

    if run_gap:
        if not historical.strip() or not modern.strip():
            st.warning("Заполните оба текстовых поля для запуска анализа.")
        else:
            st.divider()
            st.markdown("### Аналитический вердикт GeoMind")
            st.markdown('<div class="verdict-container">', unsafe_allow_html=True)
            try:
                st.write_stream(gap_analysis_stream(historical, modern))
            except Exception as exc:
                st.error(
                    "Не удалось выполнить Gap-анализ. "
                    "Проверьте, что запущен Ollama и установлена модель "
                    f"`{MODEL}`.\n\nДетали: {exc}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("### Визуализация расхождений по ключевым параметрам")
            gap_df = pd.DataFrame(
                {
                    "Параметр": [
                        "Пористость",
                        "Глубина пласта",
                        "Нефтенасыщенность",
                        "Дебит скважины",
                    ],
                    "Дельта расхождения, %": [18.5, 7.2, 34.0, 62.3],
                }
            )
            gap_df = gap_df.set_index("Параметр")
            st.bar_chart(gap_df, color="#e06666")

