import streamlit as st

from verifier import analyze, format_report_text, report_to_json

MODEL_CHOICES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

DEFAULT_THRESHOLD = 0.55
THRESHOLD_MIN = 0.35
THRESHOLD_MAX = 0.85
THRESHOLD_STEP = 0.02


def main() -> None:
    st.set_page_config(page_title="Requirement Verifier", layout="centered")

    st.title("Верификация требований по UML-диаграмме")
    st.write(
        "Загрузите **TXT** с требованиями и **XML** с диаграммой "
        "(mxCell / draw.io / Archi export). "
        "Проверка покрытия выполняется на основе семантического сопоставления."
    )

    req_file = st.file_uploader("Требования (TXT)", type=["txt"])
    xml_file = st.file_uploader("Диаграмма (XML)", type=["xml"])

    st.subheader("Настройки")

    model_name = st.selectbox(
        "Модель эмбеддингов",
        MODEL_CHOICES,
        index=0,
        help=(
            "MiniLM — быстрее и легче. "
            "mpnet-base — обычно точнее, но требует больше ресурсов."
        ),
    )

    threshold = st.slider(
        "Порог семантической схожести",
        min_value=THRESHOLD_MIN,
        max_value=THRESHOLD_MAX,
        value=DEFAULT_THRESHOLD,
        step=THRESHOLD_STEP,
        help=(
            "Чем выше порог, тем строже проверка покрытия требований. "
            "Рекомендуемый диапазон: 0.50–0.65."
        ),
    )

    run = st.button("Проверить", type="primary", disabled=not (req_file and xml_file))
    if not run:
        return

    try:
        requirements_text = req_file.read().decode("utf-8", errors="replace")
        xml_bytes = xml_file.read()

        with st.spinner("Выполняется анализ…"):
            report = analyze(
                requirements_text=requirements_text,
                xml_bytes=xml_bytes,
                threshold=threshold,
                model_name=model_name,
                include_actors_in_coverage=True,  # всегда включено
            )

        st.subheader("Кратко")
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего требований", report.total_requirements)
        col2.metric("Покрыто", report.covered_requirements)
        col3.metric("Не покрыто", len(report.missing_requirements))

        report_text = format_report_text(report, include_debug=False)

        st.subheader("Экспорт")
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Скачать отчёт (TXT)",
            data=report_text.encode("utf-8"),
            file_name="report.txt",
            mime="text/plain; charset=utf-8",
        )
        dl2.download_button(
            "Скачать отчёт (JSON)",
            data=report_to_json(report).encode("utf-8"),
            file_name="report.json",
            mime="application/json; charset=utf-8",
        )

        st.divider()

        st.subheader("Подробный отчёт")
        st.code(report_text, language="text")

    except OSError as e:
        st.error(f"Ошибка доступа к файлам: {e}")
    except Exception as e:
        st.error(f"Неожиданная ошибка: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
