import json

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.vectorstore import similarity_search


QUIZ_SYSTEM_PROMPT = """Kamu adalah pembuat kuis untuk tujuan edukasi.
Berdasarkan konteks dari materi belajar, buat soal kuis yang menguji pemahaman terhadap konten.

Aturan:
1. Buat soal HANYA berdasarkan konteks yang diberikan — jangan mengarang fakta.
2. Campurkan jenis soal sesuai permintaan (pilihan ganda, benar/salah, jawaban singkat).
3. Untuk pilihan ganda, selalu berikan 4 opsi (A, B, C, D) dengan tepat satu jawaban benar.
4. Berikan jawaban yang benar dan penjelasan singkat untuk setiap soal.
5. Soal harus menguji pemahaman, bukan sekadar hafalan.

Kembalikan respons dalam format JSON berikut:
{
  "questions": [
    {
      "type": "multiple_choice" | "true_false" | "short_answer",
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "...",
      "explanation": "..."
    }
  ]
}

Kembalikan HANYA JSON yang valid, tanpa markdown code fences atau teks tambahan.
"""


def generate_quiz(topic: str, num_questions: int = 5, question_type: str = "mixed") -> dict:
    """Generate quiz questions from document context."""
    if not st.session_state.gemini_api_key:
        return {"error": "Gemini API key belum dikonfigurasi."}

    results = similarity_search(topic, k=6)
    relevant_docs = [(doc, score) for doc, score in results if score < 1.5]

    if not relevant_docs:
        return {"error": "Tidak ditemukan dokumen relevan untuk topik ini. Silakan unggah materi terlebih dahulu."}

    context = ""
    sources = set()
    for doc, _ in relevant_docs:
        context += doc.page_content + "\n\n"
        sources.add(doc.metadata.get("source", "unknown"))

    type_instruction = ""
    if question_type == "multiple_choice":
        type_instruction = "Buat HANYA soal pilihan ganda."
    elif question_type == "true_false":
        type_instruction = "Buat HANYA soal benar/salah."
    elif question_type == "short_answer":
        type_instruction = "Buat HANYA soal jawaban singkat."
    else:
        type_instruction = "Campurkan berbagai jenis soal (pilihan ganda, benar/salah, dan jawaban singkat)."

    user_prompt = f"""Berdasarkan konteks materi belajar berikut, buat {num_questions} soal kuis tentang "{topic}".

{type_instruction}

Konteks dari dokumen:
---
{context}
---

Buat {num_questions} soal sekarang."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=st.session_state.gemini_api_key,
            temperature=0.4,
        )

        messages = [
            SystemMessage(content=QUIZ_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        quiz_data = json.loads(content)
        quiz_data["sources"] = list(sources)
        return quiz_data

    except json.JSONDecodeError:
        return {"error": "Gagal memproses respons kuis. Silakan coba lagi."}
    except Exception as e:
        return {"error": f"Pembuatan kuis gagal: {e}"}


@st.dialog("Pembuat Kuis", width="large")
def render_quiz_dialog():
    """Dialog input untuk membuat kuis."""

    st.markdown("Buat soal kuis dari materi belajar yang telah diunggah untuk menguji pemahamanmu.")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input(
            "Topik",
            placeholder="contoh: fotosintesis, machine learning, Perang Dunia II",
            help="Subjek apa yang harus dicakup kuis?",
        )
    with col2:
        num_questions = st.slider("Jumlah soal", min_value=3, max_value=15, value=5)

    question_type = st.selectbox(
        "Jenis soal",
        options=["mixed", "multiple_choice", "true_false", "short_answer"],
        format_func=lambda x: {
            "mixed": "Campuran (semua jenis)",
            "multiple_choice": "Pilihan Ganda",
            "true_false": "Benar / Salah",
            "short_answer": "Jawaban Singkat",
        }[x],
    )

    col_gen, col_close = st.columns(2)

    with col_gen:
        generate = st.button("Buat Kuis", type="primary", use_container_width=True, disabled=not topic)

    with col_close:
        if st.button("Tutup", use_container_width=True):
            st.session_state.show_quiz = False
            st.rerun()

    if generate and topic:
        with st.spinner("Membuat soal kuis dari dokumenmu..."):
            result = generate_quiz(topic, num_questions, question_type)

        if "error" in result:
            st.error(result["error"])
        else:
            # Store quiz result and open the result dialog
            st.session_state.quiz_result = result
            st.session_state.quiz_show_answers = False
            st.session_state.show_quiz = False
            st.session_state.show_quiz_result = True
            st.rerun()


@st.dialog("Hasil Kuis", width="large")
def render_quiz_result_dialog():
    """Dialog untuk menampilkan hasil kuis."""

    result = st.session_state.quiz_result
    sources = result.get("sources", [])
    questions = result.get("questions", [])

    st.success(f"Berhasil membuat {len(questions)} soal dari: {', '.join(sources)}")
    st.divider()

    # Display questions
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Soal {i}. [{q['type'].replace('_', ' ').title()}]** {q['question']}")

        if q["type"] == "multiple_choice" and q.get("options"):
            for opt in q["options"]:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{opt}")

        st.markdown("")

    st.divider()

    # Toggle answers
    if st.button("Tampilkan Jawaban" if not st.session_state.quiz_show_answers else "Sembunyikan Jawaban"):
        st.session_state.quiz_show_answers = not st.session_state.quiz_show_answers
        st.rerun()

    if st.session_state.quiz_show_answers:
        st.divider()
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Jawaban Soal {i}:** {q['answer']}")
            st.caption(f"Penjelasan: {q['explanation']}")

    st.divider()
    if st.button("Tutup"):
        st.session_state.show_quiz_result = False
        st.session_state.quiz_result = None
        st.rerun()
