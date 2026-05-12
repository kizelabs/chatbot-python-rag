import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.vectorstore import similarity_search


SUMMARY_SYSTEM_PROMPT = """Kamu adalah pembuat catatan rangkuman. Tugasmu adalah membuat catatan rangkuman
yang jelas dan terstruktur dari konteks materi belajar yang diberikan.

Aturan:
1. Hanya rangkum informasi yang ada dalam konteks yang diberikan — jangan menambahkan fakta eksternal.
2. Gunakan format yang diminta (poin-poin, catatan cornell, kerangka mind map, atau detail).
3. Sorot istilah kunci dengan huruf tebal.
4. Kelompokkan konsep terkait di bawah heading yang jelas.
5. Sertakan definisi penting, rumus, atau tanggal jika ada.
6. Jaga catatan tetap ringkas namun komprehensif — bertujuan untuk review mudah sebelum ujian.
7. Di akhir, tambahkan bagian "Poin Penting" dengan 3-5 poin utama.
"""

FORMAT_INSTRUCTIONS = {
    "bullet": "Format sebagai poin-poin hierarkis dengan topik utama dan sub-poin.",
    "cornell": (
        "Format menggunakan metode Catatan Cornell:\n"
        "- Kolom kiri: Pertanyaan/petunjuk kunci\n"
        "- Kolom kanan: Catatan detail\n"
        "- Bawah: Paragraf ringkasan\n"
        "Gunakan tabel markdown atau pemisah bagian yang jelas."
    ),
    "mindmap": (
        "Format sebagai kerangka mind map:\n"
        "- Topik sentral di atas\n"
        "- Cabang utama sebagai heading ##\n"
        "- Sub-cabang sebagai poin-poin berindentasi\n"
        "- Tunjukkan koneksi antar konsep terkait dengan catatan"
    ),
    "detailed": (
        "Format sebagai catatan belajar detail dengan:\n"
        "- Heading bagian untuk setiap topik utama\n"
        "- Penjelasan paragraf lengkap\n"
        "- Contoh jika tersedia\n"
        "- Definisi dalam blockquote"
    ),
}


def generate_summary(topic: str, note_format: str = "bullet", detail_level: str = "medium") -> dict:
    """
    Generate structured summary notes from document context.

    Args:
        topic: The topic/subject to summarize
        note_format: "bullet", "cornell", "mindmap", or "detailed"
        detail_level: "brief", "medium", or "comprehensive"

    Returns:
        Dict with 'notes' string and 'sources' list, or 'error' string
    """
    if not st.session_state.gemini_api_key:
        return {"error": "Gemini API key belum dikonfigurasi."}

    # Retrieve relevant context from vector store
    k = {"brief": 4, "medium": 6, "comprehensive": 10}.get(detail_level, 6)
    results = similarity_search(topic, k=k)
    relevant_docs = [(doc, score) for doc, score in results if score < 1.5]

    if not relevant_docs:
        return {"error": "Tidak ditemukan dokumen relevan untuk topik ini. Silakan unggah materi terlebih dahulu."}

    # Build context
    context = ""
    sources = set()
    for doc, _ in relevant_docs:
        context += doc.page_content + "\n\n"
        sources.add(doc.metadata.get("source", "unknown"))

    # Build prompt
    detail_instruction = {
        "brief": "Buat ringkas — fokus pada poin terpenting saja (target ~200 kata).",
        "medium": "Seimbangkan antara keringkasan dan detail (target ~500 kata).",
        "comprehensive": "Buat menyeluruh — bahas semua poin dalam konteks dengan penjelasan lengkap (target ~1000 kata).",
    }.get(detail_level, "")

    format_instruction = FORMAT_INSTRUCTIONS.get(note_format, FORMAT_INSTRUCTIONS["bullet"])

    user_prompt = f"""Buat catatan rangkuman tentang "{topic}" dari materi berikut.

Format: {format_instruction}

Tingkat detail: {detail_instruction}

Konteks dari dokumen:
---
{context}
---

Buat catatan rangkuman sekarang."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=st.session_state.gemini_api_key,
            temperature=0.3,
        )

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        return {
            "notes": response.content,
            "sources": list(sources),
        }

    except Exception as e:
        return {"error": f"Pembuatan rangkuman gagal: {e}"}


@st.dialog("Pembuat Rangkuman", width="large")
def render_summary_dialog():
    """Render the summary notes generator dialog."""

    st.markdown("Buat catatan rangkuman terstruktur dari materi yang telah diunggah untuk review cepat.")

    topic = st.text_input(
        "Topik",
        placeholder="contoh: biologi sel, jaringan saraf, Revolusi Perancis",
        help="Subjek apa yang harus dirangkum?",
    )

    col1, col2 = st.columns(2)
    with col1:
        note_format = st.selectbox(
            "Format catatan",
            options=["bullet", "cornell", "mindmap", "detailed"],
            format_func=lambda x: {
                "bullet": "Poin-poin",
                "cornell": "Catatan Cornell",
                "mindmap": "Kerangka Mind Map",
                "detailed": "Paragraf Detail",
            }[x],
        )
    with col2:
        detail_level = st.selectbox(
            "Tingkat detail",
            options=["brief", "medium", "comprehensive"],
            format_func=lambda x: {
                "brief": "Ringkas (~200 kata)",
                "medium": "Sedang (~500 kata)",
                "comprehensive": "Lengkap (~1000 kata)",
            }[x],
            index=1,
        )

    col_gen, col_close = st.columns(2)

    with col_gen:
        generate = st.button("Buat Rangkuman", type="primary", use_container_width=True, disabled=not topic)

    with col_close:
        if st.button("Tutup", use_container_width=True):
            st.session_state.show_summary = False
            st.rerun()

    if generate and topic:
        with st.spinner("Membuat rangkuman dari dokumenmu..."):
            result = generate_summary(topic, note_format, detail_level)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Rangkuman dibuat dari: {', '.join(result['sources'])}")
            st.divider()
            st.markdown(result["notes"])

            st.divider()
            # Download button
            st.download_button(
                label="Unduh sebagai Markdown",
                data=result["notes"],
                file_name=f"rangkuman_{topic.replace(' ', '_')}.md",
                mime="text/markdown",
            )
