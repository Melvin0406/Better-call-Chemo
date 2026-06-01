import os

import streamlit as st
from dotenv import load_dotenv

from rag import Assistant


def load_environment_config() -> dict[str, str | None]:
    load_dotenv()
    return {
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or None,
        "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or None,
        "model": os.getenv("LLM_MODEL") or os.getenv("MODEL") or None,
    }


def init_assistant() -> Assistant:
    config = load_environment_config()
    try:
        return Assistant.from_config(config)
    except FileNotFoundError:
        st.error(
            "El índice de documentos no existe. "
            "Ejecuta `python ingest.py` en la terminal y recarga la página."
        )
        st.stop()
    except Exception as exc:
        st.error(f"No se pudo inicializar el asistente: {exc}")
        st.stop()


st.set_page_config(
    page_title="Better Call Chemo",
    page_icon="⚖️",
    layout="wide",
)


def main() -> None:
    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚖️ Better Call Chemo")
        st.caption("Asistente legal y fiscal mexicano")
        st.markdown("---")

        st.markdown("### ¿Qué puedo preguntarle?")
        st.markdown(
            "Consulta sobre leyes federales mexicanas: ISR, IVA, "
            "derecho laboral, penal, civil, mercantil y más. "
            "Las respuestas se basan en los textos oficiales de los documentos indexados."
        )

        st.markdown("---")
        st.markdown("### Ejemplos de preguntas")
        st.markdown(
            "- ¿Cuáles son mis deducciones personales como asalariado?\n"
            "- ¿Qué pasa si no presento mi declaración anual?\n"
            "- ¿Cuándo estoy obligado a cobrar IVA?\n"
            "- ¿Qué derechos tengo si me despiden sin causa justificada?\n"
            "- ¿Qué es el RFC y cómo se tramita?"
        )

        st.markdown("---")
        st.markdown("### Filtros disponibles")
        st.markdown(
            "Agrega un filtro al final de tu pregunta para buscar en un área o documento específico:\n\n"
            "**Por área:**  \n"
            "`/fiscal` `/laboral` `/penal` `/civil`  \n"
            "`/mercantil` `/administrativo` `/constitucional`\n\n"
            "**Por documento:**  \n"
            "`/cff` `/lisr` `/liva` `/rmf` `/lft` `/cpf`"
        )

        st.markdown("---")
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.assistant.clear_history()
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption(
            "Las respuestas son orientativas y no constituyen asesoría legal formal. "
            "Consulta con un abogado o contador para casos concretos."
        )

    # ── Inicializar estado ─────────────────────────────────────────────────────
    if "assistant" not in st.session_state:
        with st.spinner("Cargando documentos..."):
            st.session_state.assistant = init_assistant()
        st.session_state.messages = []

    assistant: Assistant = st.session_state.assistant

    # ── Historial de mensajes ──────────────────────────────────────────────────
    CHEMO_AVATAR = "chemo.png"

    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=CHEMO_AVATAR):
            st.markdown(
                "Hola, soy **Chemo**, tu asistente legal y fiscal mexicano. "
                "Puedo ayudarte a consultar el Código Fiscal, la Ley del ISR, "
                "la Ley Federal del Trabajo y muchas otras leyes federales. "
                "\n\n¿En qué te puedo ayudar hoy?"
            )

    for msg in st.session_state.messages:
        avatar = CHEMO_AVATAR if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Input del usuario ──────────────────────────────────────────────────────
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar=CHEMO_AVATAR):
            with st.spinner("Consultando documentos..."):
                answer = assistant.ask(prompt)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
