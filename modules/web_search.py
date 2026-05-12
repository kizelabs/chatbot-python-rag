import streamlit as st
from exa_py import Exa


def search_web(query: str, num_results: int = 3) -> list[dict]:
    """
    Search the web using Exa API.
    Returns a list of dicts with 'title', 'url', and 'text' keys.
    """
    if not st.session_state.exa_api_key:
        return []

    try:
        exa = Exa(api_key=st.session_state.exa_api_key)
        results = exa.search_and_contents(
            query,
            type="auto",
            use_autoprompt=True,
            num_results=num_results,
            text={"max_characters": 1500},
        )

        search_results = []
        for result in results.results:
            search_results.append(
                {
                    "title": result.title or "Tanpa Judul",
                    "url": result.url,
                    "text": result.text or "",
                }
            )
        return search_results

    except Exception as e:
        st.warning(f"Pencarian web gagal: {e}")
        return []


def format_web_results(results: list[dict]) -> str:
    """Format web search results into a context string for the LLM."""
    if not results:
        return ""

    formatted = "**Hasil Pencarian Web:**\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"[{i}] {result['title']}\n"
        formatted += f"    URL: {result['url']}\n"
        formatted += f"    {result['text'][:500]}...\n\n"

    return formatted
