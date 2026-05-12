import streamlit as st
from modules.storage import save_to_localstorage


PERSONA_OPTIONS = {
    "friendly": "Ramah & Menyemangati — menjelaskan seperti tutor yang suportif",
    "academic": "Akademis & Presisi — formal, fokus pada kutipan dan referensi",
    "socratic": "Sokratik — menjawab dengan pertanyaan penuntun untuk memperdalam pemahaman",
}


@st.dialog("Pengaturan", width="large")
def render_settings_dialog():
    """Render the settings dialog with API keys and parameters."""

    st.markdown("#### API Key")
    st.caption("Key disimpan di browser (localStorage) agar tidak perlu diisi ulang.")

    gemini_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="AIza...",
        help="Digunakan untuk LLM (chat, kuis, rangkuman)",
    )

    nvidia_key = st.text_input(
        "NVIDIA NIM API Key",
        value=st.session_state.nvidia_api_key,
        type="password",
        placeholder="nvapi-...",
        help="Digunakan untuk embedding dokumen",
    )

    exa_key = st.text_input(
        "Exa API Key",
        value=st.session_state.exa_api_key,
        type="password",
        placeholder="exa-...",
        help="Digunakan untuk pencarian web sebagai cadangan",
    )

    exa_enabled = st.toggle(
        "Aktifkan pencarian web (Exa)",
        value=st.session_state.exa_enabled,
        help="Jika diaktifkan, Baim akan mencari di web ketika jawaban tidak ditemukan di dokumen",
    )

    st.divider()
    st.markdown("#### Parameter Model")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lebih tinggi = lebih kreatif, Lebih rendah = lebih fokus",
    )

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=100,
        value=st.session_state.top_k,
        step=1,
        help="Jumlah token teratas yang dipertimbangkan saat generasi",
    )

    st.divider()
    st.markdown("#### Persona")

    persona = st.selectbox(
        "Gaya Mengajar",
        options=list(PERSONA_OPTIONS.keys()),
        format_func=lambda x: PERSONA_OPTIONS[x],
        index=list(PERSONA_OPTIONS.keys()).index(st.session_state.persona),
        help="Pilih bagaimana Baim merespons pertanyaanmu",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simpan", type="primary", use_container_width=True):
            st.session_state.gemini_api_key = gemini_key
            st.session_state.nvidia_api_key = nvidia_key
            st.session_state.exa_api_key = exa_key
            st.session_state.exa_enabled = exa_enabled
            st.session_state.temperature = temperature
            st.session_state.top_k = top_k
            st.session_state.persona = persona
            save_to_localstorage()
            st.session_state.show_settings = False
            st.rerun()

    with col2:
        if st.button("Batal", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()
