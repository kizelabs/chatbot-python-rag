"""LocalStorage persistence for settings using streamlit-javascript."""

import json
import streamlit as st
import streamlit.components.v1 as components


STORAGE_KEY = "baim_ai_settings"

# Settings keys to persist
PERSIST_KEYS = ["gemini_api_key", "nvidia_api_key", "exa_api_key", "exa_enabled", "temperature", "top_k", "persona"]


def save_to_localstorage():
    """Save current settings to browser localStorage via injected JS."""
    settings = {}
    for key in PERSIST_KEYS:
        if key in st.session_state:
            settings[key] = st.session_state[key]

    settings_json = json.dumps(settings)
    components.html(
        f"""<script>
        localStorage.setItem('{STORAGE_KEY}', JSON.stringify({settings_json}));
        </script>""",
        height=0,
        width=0,
    )


def load_from_localstorage():
    """
    Load settings from localStorage on page load.
    Injects JS that reads localStorage and sends data back to Streamlit
    via a hidden iframe with query parameters, triggering a one-time reload.
    """
    if st.session_state.get("_storage_restored"):
        return

    # Check if we have settings passed via fragment
    fragment = st.query_params.get("_s", None)
    if fragment:
        try:
            import base64
            decoded = base64.b64decode(fragment).decode("utf-8")
            settings = json.loads(decoded)
            for key in PERSIST_KEYS:
                if key in settings and settings[key] != "":
                    st.session_state[key] = settings[key]
            st.session_state["_storage_restored"] = True
            # Clear the query param
            st.query_params.clear()
            return
        except Exception:
            st.session_state["_storage_restored"] = True
            st.query_params.clear()
            return

    # First visit: inject JS to read localStorage and redirect with data
    components.html(
        f"""<script>
        (function() {{
            const stored = localStorage.getItem('{STORAGE_KEY}');
            if (stored && stored !== '{{}}') {{
                const encoded = btoa(stored);
                const url = new URL(window.parent.location.href);
                if (!url.searchParams.has('_s')) {{
                    url.searchParams.set('_s', encoded);
                    window.parent.location.href = url.toString();
                }}
            }}
        }})();
        </script>""",
        height=0,
        width=0,
    )
    st.session_state["_storage_restored"] = True
