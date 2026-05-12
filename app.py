"""
Baim AI - Asisten Belajar Pribadi
Chatbot berbasis RAG yang membantu siswa belajar dari dokumen mereka.
Menggunakan Gemini sebagai LLM, PGVector untuk penyimpanan, dan Exa untuk pencarian web.
"""

import streamlit as st
from modules.chat import render_chat
from modules.settings import render_settings_dialog
from modules.ingest import render_ingest_dialog
from modules.tools.quiz_generator import render_quiz_dialog, render_quiz_result_dialog
from modules.tools.summary_notes import render_summary_dialog
from modules.storage import load_from_localstorage


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "gemini_api_key": "",
        "nvidia_api_key": "",
        "exa_api_key": "",
        "exa_enabled": False,
        "temperature": 0.7,
        "top_k": 40,
        "show_settings": False,
        "show_ingest": False,
        "show_quiz": False,
        "show_quiz_result": False,
        "quiz_result": None,
        "quiz_show_answers": False,
        "show_summary": False,
        "documents_ingested": 0,
        "persona": "friendly",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _close_all_dialogs():
    """Reset all dialog flags to ensure only one opens at a time."""
    st.session_state.show_settings = False
    st.session_state.show_ingest = False
    st.session_state.show_quiz = False
    st.session_state.show_quiz_result = False
    st.session_state.show_summary = False


def main():
    st.set_page_config(
        page_title="Baim AI",
        page_icon="B",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }
        .header-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
        }
        .header-subtitle {
            font-size: 0.875rem;
            color: #64748b;
        }
        .doc-badge {
            background: #f0fdf4;
            color: #166534;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        /* Buttons: prevent text wrapping */
        .stButton > button {
            white-space: nowrap !important;
            font-size: 0.8rem !important;
            padding: 0.4rem 0.6rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Restore settings from localStorage on first load
    load_from_localstorage()

    # Header
    col1, col2 = st.columns([5, 5])

    with col1:
        st.markdown("### Baim AI")
        st.caption("Asisten belajar pribadimu — tanya apa saja tentang dokumenmu")

    with col2:
        btn_cols = st.columns([1, 1, 1, 1], gap="small")
        with btn_cols[0]:
            if st.button("📄 Unggah", use_container_width=True):
                _close_all_dialogs()
                st.session_state.show_ingest = True
                st.rerun()
        with btn_cols[1]:
            if st.button("📝 Kuis", use_container_width=True):
                _close_all_dialogs()
                st.session_state.show_quiz = True
                st.rerun()
        with btn_cols[2]:
            if st.button("📋 Rangkum", use_container_width=True):
                _close_all_dialogs()
                st.session_state.show_summary = True
                st.rerun()
        with btn_cols[3]:
            if st.button("⚙️ Setelan", use_container_width=True):
                _close_all_dialogs()
                st.session_state.show_settings = True
                st.rerun()

    if st.session_state.documents_ingested > 0:
        st.markdown(
            f'<span class="doc-badge">{st.session_state.documents_ingested} dokumen dimuat</span>',
            unsafe_allow_html=True,
        )

    # Dialogs (only one at a time)
    if st.session_state.show_settings:
        render_settings_dialog()
    elif st.session_state.show_ingest:
        render_ingest_dialog()
    elif st.session_state.show_quiz:
        render_quiz_dialog()
    elif st.session_state.show_quiz_result and st.session_state.quiz_result:
        render_quiz_result_dialog()
    elif st.session_state.show_summary:
        render_summary_dialog()

    # Main chat area
    render_chat()


if __name__ == "__main__":
    main()
