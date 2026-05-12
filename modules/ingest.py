import os
import base64
import tempfile

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from modules.vectorstore import get_vectorstore


SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"]

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def extract_text_from_image(file_path: str) -> str:
    """Use Gemini Vision to extract text/content from an image."""
    with open(file_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = file_path.rsplit(".", 1)[-1].lower()
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=st.session_state.gemini_api_key,
    )

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Ekstrak semua teks dan informasi dari gambar ini. Jika ada diagram, tabel, atau rumus, jelaskan secara detail. Kembalikan hasilnya sebagai teks biasa."},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
        ],
    }

    response = llm.invoke([message])
    return response.content


def load_document(file_path: str, file_type: str):
    """Load a document based on its file type."""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()
    elif file_type in IMAGE_EXTENSIONS:
        text = extract_text_from_image(file_path)
        return [Document(page_content=text, metadata={"type": "image"})]
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


@st.dialog("Unggah Dokumen", width="large")
def render_ingest_dialog():
    """Render the document upload and ingestion dialog."""

    st.markdown("Unggah materi belajarmu (PDF, DOCX, TXT, Gambar) untuk membangun basis pengetahuan.")

    uploaded_files = st.file_uploader(
        "Pilih file",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Unggah satu atau lebih dokumen untuk diproses ke basis pengetahuan",
    )

    col_size, col_overlap = st.columns(2)
    with col_size:
        chunk_size = st.number_input("Ukuran chunk", min_value=200, max_value=4000, value=1000, step=100)
    with col_overlap:
        chunk_overlap = st.number_input("Overlap chunk", min_value=0, max_value=500, value=200, step=50)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Proses", type="primary", use_container_width=True, disabled=not uploaded_files):
            if not st.session_state.nvidia_api_key:
                st.error("Silakan atur NVIDIA NIM API key di Pengaturan terlebih dahulu.")
                return

            progress = st.progress(0, text="Memproses dokumen...")
            all_chunks = []

            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Memproses {uploaded_file.name}...",
                )

                # Save to temp file
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    docs = load_document(tmp_path, suffix.lstrip("."))
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    all_chunks.extend(chunks)
                finally:
                    os.unlink(tmp_path)

            if all_chunks:
                progress.progress(0.8, text="Menyimpan embedding ke database vektor...")
                try:
                    vectorstore = get_vectorstore()
                    vectorstore.add_documents(all_chunks)
                    st.session_state.documents_ingested += len(uploaded_files)
                    progress.progress(1.0, text="Selesai!")
                    st.success(
                        f"Berhasil memproses {len(uploaded_files)} file — {len(all_chunks)} chunk tersimpan."
                    )
                except Exception as e:
                    st.error(f"Gagal menyimpan embedding: {e}")
            else:
                st.warning("Tidak ada konten yang berhasil diekstrak dari file yang diunggah.")

    with col2:
        if st.button("Tutup", use_container_width=True):
            st.session_state.show_ingest = False
            st.rerun()
