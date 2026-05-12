import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from modules.vectorstore import similarity_search
from modules.web_search import search_web, format_web_results


PERSONA_PROMPTS = {
    "friendly": (
        "Kamu adalah Baim, asisten belajar yang ramah dan penuh semangat. "
        "Kamu menjelaskan konsep dengan jelas menggunakan analogi dan contoh. "
        "Kamu merayakan ketika siswa memahami sesuatu dan dengan lembut mengoreksi kesalahan. "
        "Jaga nada bicaramu tetap hangat, suportif, dan santai."
    ),
    "academic": (
        "Kamu adalah Baim, asisten akademik yang fokus pada ketepatan dan ketelitian. "
        "Kamu memberikan jawaban yang terstruktur dengan terminologi yang tepat. "
        "Ketika merujuk informasi dari dokumen, cantumkan sumbernya. "
        "Pertahankan nada formal namun tetap mudah didekati."
    ),
    "socratic": (
        "Kamu adalah Baim, asisten belajar dengan metode Sokratik. "
        "Alih-alih memberikan jawaban langsung, bimbing siswa dengan pertanyaan yang mendalam. "
        "Bantu mereka menemukan jawaban sendiri melalui penalaran. "
        "Berikan jawaban langsung hanya jika siswa jelas-jelas kesulitan setelah beberapa kali mencoba."
    ),
}

SYSTEM_TEMPLATE = """{persona_prompt}

Kamu membantu siswa belajar dari materi studi yang mereka unggah. Saat menjawab:
1. Pertama, gunakan konteks yang disediakan dari dokumen mereka untuk menjawab pertanyaan.
2. Jika konteks tidak mengandung informasi yang relevan, gunakan hasil pencarian web jika tersedia.
3. Jika kedua sumber tidak memiliki jawabannya, katakan dengan jujur dan sarankan apa yang bisa mereka cari.
4. Gunakan riwayat percakapan untuk memahami pertanyaan lanjutan dan menjaga kesinambungan.

Selalu tunjukkan sumbermu:
- Dari dokumenmu: [nama file sumber]
- Dari pencarian web: [URL sumber]
- Pengetahuan umum (ketika kedua sumber tidak berlaku)

{context_section}
"""

REWRITE_PROMPT = """Berdasarkan riwayat percakapan dan pertanyaan terbaru pengguna, tulis ulang pertanyaan tersebut
menjadi kueri pencarian mandiri yang menangkap maksud lengkapnya. Sertakan konteks relevan dari
percakapan agar kueri dapat digunakan untuk pencarian dokumen tanpa riwayat chat.

Jika pertanyaan terbaru sudah cukup jelas dengan sendirinya, kembalikan apa adanya.
Kembalikan HANYA kueri yang ditulis ulang, tidak ada yang lain.

Riwayat percakapan:
{history}

Pertanyaan terbaru: {question}

Kueri pencarian mandiri:"""


def rewrite_query_with_history(query: str) -> str:
    """
    Use chat history to rewrite ambiguous follow-up questions into standalone queries.
    E.g., "explain more" → "explain more about photosynthesis light reactions"
    """
    # If no history or query is already detailed (>50 chars), skip rewriting
    if not st.session_state.messages or len(query) > 80:
        return query

    # Only rewrite if the query looks like a follow-up (short, uses pronouns, etc.)
    follow_up_indicators = [
        "it", "that", "this", "those", "these", "them",
        "more", "again", "elaborate", "explain", "why",
        "how", "what about", "and", "also", "continue",
    ]
    query_lower = query.lower()
    is_follow_up = len(query.split()) <= 8 or any(
        indicator in query_lower for indicator in follow_up_indicators
    )

    if not is_follow_up:
        return query

    # Build recent history summary (last 4 messages)
    recent = st.session_state.messages[-4:]
    history_text = ""
    for msg in recent:
        role = "Siswa" if msg["role"] == "user" else "Asisten"
        # Potong pesan asisten yang terlalu panjang
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        history_text += f"{role}: {content}\n"

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=st.session_state.gemini_api_key,
            temperature=0.0,
        )
        prompt = REWRITE_PROMPT.format(history=history_text, question=query)
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        # Sanity check — don't use if it's too long or empty
        if rewritten and len(rewritten) < 300:
            return rewritten
    except Exception:
        pass

    return query


def build_context(query: str) -> tuple[str, bool]:
    """
    Build context by searching vector store first, then web if needed.
    Uses chat history to improve retrieval for follow-up questions.
    Returns (context_string, used_web_search).
    """
    # Rewrite query using conversation history for better retrieval
    search_query = rewrite_query_with_history(query)

    # Step 1: Search vector store
    doc_results = similarity_search(search_query, k=4)

    # Filter by relevance score (lower = more similar for L2 distance)
    relevant_docs = [(doc, score) for doc, score in doc_results if score < 1.5]

    if relevant_docs:
        context = "**Konteks dari dokumenmu:**\n\n"
        for doc, score in relevant_docs:
            source = doc.metadata.get("source", "tidak diketahui")
            context += f"[Sumber: {source}]\n{doc.page_content}\n\n"
        return context, False

    # Step 2: Fallback to web search
    if st.session_state.exa_enabled and st.session_state.exa_api_key:
        web_results = search_web(search_query)
        if web_results:
            return format_web_results(web_results), True

    return "", False


def get_llm():
    """Create the Gemini LLM instance with current settings."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=st.session_state.gemini_api_key,
        temperature=st.session_state.temperature,
        top_k=st.session_state.top_k,
        streaming=True,
    )


def generate_response(query: str):
    """Generate a streaming response using RAG pipeline."""
    # Build context (uses history-aware query rewriting)
    context, used_web = build_context(query)

    if context:
        context_section = f"Gunakan konteks berikut untuk menjawab pertanyaan:\n\n{context}"
    else:
        context_section = "Tidak ditemukan konteks relevan dari dokumen atau pencarian web. Jawab berdasarkan pengetahuan umummu dan nyatakan dengan jelas."

    # Build system message
    persona_prompt = PERSONA_PROMPTS.get(st.session_state.persona, PERSONA_PROMPTS["friendly"])
    system_content = SYSTEM_TEMPLATE.format(
        persona_prompt=persona_prompt,
        context_section=context_section,
    )

    # Build message history
    messages = [SystemMessage(content=system_content)]

    # Add full conversation history (last 20 messages for better continuity)
    for msg in st.session_state.messages[-20:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Add current query
    messages.append(HumanMessage(content=query))

    # Generate with streaming
    llm = get_llm()
    return llm.stream(messages)


def render_chat():
    """Render the chat interface."""

    # Inject thinking animation CSS
    st.markdown(
        """
        <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .thinking-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 0;
            color: #6366f1;
            font-size: 0.9rem;
            font-weight: 500;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .thinking-indicator .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #6366f1;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .thinking-indicator .dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-indicator .dot:nth-child(3) { animation-delay: 0.4s; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Check if API keys are configured
    if not st.session_state.gemini_api_key:
        st.info(
            "Selamat datang di Baim AI! Untuk memulai:\n"
            "1. Klik tombol Setelan untuk mengatur API key (Gemini wajib diisi)\n"
            "2. Klik tombol Unggah untuk mengunggah materi belajarmu\n"
            "3. Mulai bertanya!"
        )
        return

    # Handle retry
    if st.session_state.get("retry_prompt"):
        prompt = st.session_state.retry_prompt
        st.session_state.retry_prompt = None
        # Remove the last failed assistant message
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
        _process_query(prompt)
        st.rerun()
        return

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show retry button on the last message if it was an error
            if (
                message["role"] == "assistant"
                and i == len(st.session_state.messages) - 1
                and message.get("is_error")
            ):
                if st.button("Coba Lagi", key=f"retry_{i}"):
                    # Find the last user message to retry
                    for msg in reversed(st.session_state.messages[:-1]):
                        if msg["role"] == "user":
                            st.session_state.retry_prompt = msg["content"]
                            st.rerun()
                            break

    # Chat input
    if prompt := st.chat_input("Tanya apa saja tentang materi belajarmu..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        _process_query(prompt)


def _process_query(prompt: str):
    """Process a user query — show thinking, generate response, handle errors."""
    with st.chat_message("assistant"):
        # Show thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<div class="thinking-indicator">'
            '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
            '&nbsp; Sedang berpikir...</div>',
            unsafe_allow_html=True,
        )

        try:
            stream = generate_response(prompt)
            # Clear thinking indicator and stream response
            thinking_placeholder.empty()
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"Terjadi kesalahan: {e}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "is_error": True}
            )
            if st.button("Coba Lagi"):
                st.session_state.retry_prompt = prompt
                st.rerun()
