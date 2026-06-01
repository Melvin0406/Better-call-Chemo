import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import Assistant, resolve_config


def load_environment_config() -> dict[str, str | None]:
    load_dotenv()
    return {
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or None,
        "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or None,
        "model": os.getenv("LLM_MODEL") or os.getenv("MODEL") or None,
    }


def init_assistant() -> Assistant | None:
    config = load_environment_config()
    if not config["api_key"]:
        st.warning(
            "No se encontró una clave de OpenAI en las variables de entorno. "
            "Establece OPENAI_API_KEY o API_KEY antes de ejecutar la aplicación."
        )

    try:
        assistant = Assistant.from_config(config)
        return assistant
    except FileNotFoundError as exc:
        st.error(
            "No se encontró el índice FAISS. Ejecuta `python ingest.py` primero para "
            "generar los archivos de índice en la carpeta `index/`."
        )
        st.stop()
    except Exception as exc:
        st.error(f"No se pudo inicializar el asistente: {exc}")
        st.stop()


def apply_theme() -> None:
    st.set_page_config(
        page_title="Better Call Chemo",
        page_icon="none",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .css-1d391kg {
            background-color: #07142d !important;
        }
        .stApp {
            background: #031224;
            color: #e9f2ff;
        }
        .css-18e3th9 {
            background-color: #07142d;
        }
        .stButton>button {
            background-color: #1c4db7;
            color: #ffffff;
            border: 1px solid #4da6ff;
        }
        .stButton>button:hover {
            background-color: #356bc4;
        }
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div>div {
            border: 1px solid #4da6ff !important;
            box-shadow: none !important;
            background-color: #0c1a3a !important;
            color: #e9f2ff !important;
        }
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {
            border-color: #70b4ff !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(77, 166, 255, 0.2);
        }
        .css-1x8cf1d {
            background-color: #07142d !important;
        }
        .css-10trblm {
            background-color: #07142d !important;
        }
        .stAlert {
            background-color: #0b213d !important;
            border-left: 4px solid #4da6ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()

    st.title("Better Call Chemo")
    st.markdown(
        "Una interfaz RAG para consultar leyes federales mexicanas usando documentos indexados. "
        "Preguntas con filtros como `/LISR` o `/fiscal` son compatibles."
    )

    if "assistant" not in st.session_state:
        st.session_state.assistant = init_assistant()
        st.session_state.messages = []

    assistant: Assistant = st.session_state.assistant

    with st.expander("Configuración de conexión", expanded=False):
        config = load_environment_config()
        st.write(
            "Se cargó la siguiente configuración desde el entorno (solo para lectura):"
        )
        st.write({
            "OPENAI_API_KEY": bool(config["api_key"]),
            "OPENAI_BASE_URL / BASE_URL": config["base_url"],
            "LLM_MODEL / MODEL": config["model"],
        })

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Pregunta",
            value="",
            height=140,
            placeholder="Escribe tu pregunta sobre leyes federales mexicanas aquí...",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Clear Context"):
            assistant.clear_history()
            st.session_state.messages = []
            st.success("Contexto y historial de conversación borrados.")

    if st.button("Enviar"):
        if not query.strip():
            st.warning("Escribe una pregunta antes de enviar.")
        else:
            with st.spinner("Consultando al asistente..."):
                answer = assistant.ask(query)
            st.session_state.messages.append((query, answer))

    if st.session_state.messages:
        st.markdown("---")
        st.subheader("Historial de conversación")
        for user_message, assistant_message in st.session_state.messages[::-1]:
            st.markdown(f"**Tú:** {user_message}")
            st.markdown(f"**Asistente:** {assistant_message}")
            st.markdown("---")


if __name__ == "__main__":
    main()
