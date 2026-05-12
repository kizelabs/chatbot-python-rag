# Baim AI — Asisten Belajar Pribadi

Chatbot berbasis RAG (Retrieval-Augmented Generation) yang membantu siswa belajar dari dokumen yang mereka unggah. Dibangun dengan Streamlit, Gemini, LangChain, PGVector, NVIDIA NIM embeddings, dan Exa web search.

## Use Case

**Asisten Edukasi Pribadi** — Siswa mengunggah catatan kuliah, buku teks (PDF/DOCX/TXT/Gambar), dan Baim menjawab pertanyaan menggunakan materi tersebut sebagai konteks. Ketika jawaban tidak ada di dokumen, Baim mencari di web melalui Exa (jika diaktifkan).

### Parameter Kreatif

| Parameter         | Pilihan                                            |
| -------------------| ----------------------------------------------------|
| **Persona**       | Ramah & Menyemangati, Akademis & Presisi, Sokratik |
| **Temperature**   | 0.0 – 2.0 (mengontrol kreativitas)                 |
| **Top K**         | 1 – 100 (lebar sampling token)                     |
| **Ukuran Chunk**  | 200 – 4000 (granularitas pemecahan dokumen)        |
| **Pencarian Web** | Aktif / Nonaktif (toggle di pengaturan)            |

## Arsitektur

```
┌─────────────────────────────────────────────────┐
│                 Streamlit UI                    │
│  ┌──────────┐  ┌──────────┐  ┌────-──────────┐  │
│  │   Chat   │  │Pengaturan│  │    Unggah     │  │
│  │  Module  │  │  Dialog  │  │    Dialog     │  │
│  └────┬─────┘  └──────────┘  └────-──┬───────┘  │
│       │                              │          │
│  ┌────▼──────────────────────────────▼───────┐  │
│  │           LangChain RAG Pipeline          │  │
│  │                                           │  │
│  │  Query → Vector Search → Bangun Konteks   │  │
│  │       → (Pencarian Web cadangan) → LLM    │  │
│  └────┬──────────────┬───────────────┬───────┘  │
│       │              │               │          │
└───────┼──────────────┼───────────────┼──────────┘
        │              │               │
   ┌────▼────┐    ┌────▼────┐    ┌─────▼─────┐
   │ Gemini  │    │PGVector │    │   Exa     │
   │  LLM    │    │+ NVIDIA │    │Pencarian  │
   │         │    │  NIM    │    │   Web     │
   └─────────┘    └─────────┘    └───────────┘
```

## Fitur

- **Chat RAG** — Tanya jawab berdasarkan dokumen yang diunggah
- **Pembuat Kuis** — Buat soal kuis (pilihan ganda, benar/salah, jawaban singkat) dari materi
- **Pembuat Rangkuman** — Buat catatan rangkuman dengan berbagai format (poin-poin, Cornell, mind map, detail)
- **OCR Gambar** — Unggah gambar (PNG, JPG, WEBP) dan Gemini Vision mengekstrak teksnya
- **Pencarian Web** — Fallback ke Exa web search jika jawaban tidak ada di dokumen (bisa diaktifkan/nonaktifkan)
- **Riwayat Percakapan** — Query rewriting otomatis untuk pertanyaan lanjutan
- **Penyimpanan Lokal** — API key dan pengaturan disimpan di browser (localStorage)

## Mulai Cepat

### Prasyarat

- Docker & Docker Compose
- API Key:
  - [Gemini API Key](https://aistudio.google.com/apikey) (wajib — untuk LLM dan OCR gambar)
  - [NVIDIA NIM API Key](https://build.nvidia.com/) (wajib — untuk embedding dokumen)
  - [Exa API Key](https://exa.ai/) (opsional — untuk pencarian web)

### Jalankan dengan Docker

```bash
docker compose up --build
```

Buka [http://localhost:8501](http://localhost:8501)

### Jalankan Lokal (development)

```bash
# Jalankan database PGVector saja
docker compose up pgvector -d

# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

## Alur Penggunaan

1. **Konfigurasi** — Klik tombol "Pengaturan", masukkan API key dan atur parameter model
2. **Unggah** — Klik tombol "Unggah", pilih file materi belajar (PDF, DOCX, TXT, Gambar)
3. **Chat** — Tanyakan apa saja tentang dokumenmu
4. **Kuis** — Klik tombol "Kuis" untuk membuat soal latihan dari materi
5. **Rangkum** — Klik tombol "Rangkum" untuk membuat catatan ringkas

## Stack Teknologi

| Komponen       | Teknologi                               |
| ----------------| -----------------------------------------|
| Frontend       | Streamlit                               |
| LLM            | Google Gemini 2.5 Flash Lite            |
| Embeddings     | NVIDIA NIM (llama-nemotron-embed-1b-v2) |
| Vector Store   | PGVector (PostgreSQL)                   |
| Framework RAG  | LangChain                               |
| Pencarian Web  | Exa API                                 |
| Kontainerisasi | Docker Compose                          |

## Struktur Proyek

```
├── app.py                       # Aplikasi utama Streamlit (entry point)
├── modules/
│   ├── chat.py                  # Chat UI + RAG pipeline
│   ├── settings.py              # Dialog pengaturan
│   ├── ingest.py                # Upload & pemrosesan dokumen
│   ├── vectorstore.py           # PGVector + NVIDIA NIM embeddings
│   ├── web_search.py            # Pencarian web Exa (optional)
│   ├── storage.py               # Persistensi localStorage
│   └── tools/
│       ├── quiz_generator.py    # Pembuat kuis
│       └── summary_notes.py     # Pembuat rangkuman
├── docker-compose.yml           # Layanan PGVector + App
├── Dockerfile                   # Container aplikasi
├── requirements.txt             # Dependensi Python
└── .streamlit/config.toml       # Tema Streamlit
```

## Model yang Digunakan

| Fungsi            | Model                             | Provider   |
| -------------------| -----------------------------------| ------------|
| Chat & Generasi   | gemini-2.5-flash-lite             | Google     |
| Embedding Dokumen | nvidia/llama-nemotron-embed-1b-v2 | NVIDIA NIM |
| OCR Gambar        | gemini-2.5-flash-lite             | Google     |
