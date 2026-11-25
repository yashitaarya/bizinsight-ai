"""Cached data steps and shared AI settings used by several tabs.

st.cache_data remembers results for identical inputs, so clicking around the app
doesn't re-read or re-analyze the file. The cache lives in server memory only.
"""

from __future__ import annotations

import streamlit as st

from core import config
from core.data_loader import LoadedDataset, list_excel_sheets, load_dataset
from core.facts import Fact, build_fact_sheet
from core.formatting import Formatter
from core.llm_client import get_settings
from core.profiler import DatasetProfile, profile_dataset

Source = tuple[str, bytes, "str | None"]   # (file name, file bytes, sheet name)

PAGES = ["Home", "Overview", "AI Summary", "Ask AI", "Charts", "Data"]


def go_to(page: str, generate_summary: bool = False) -> None:
    """Button callback: switch page (and optionally start the AI summary when it opens)."""
    st.session_state["nav"] = page
    if generate_summary:
        st.session_state["ai_autogenerate"] = True


def summary_key(source: Source, model: str) -> str:
    """Identifies a generated summary: same file + sheet + model = same summary."""
    import hashlib
    return hashlib.sha256(source[1]).hexdigest()[:16] + f"|{source[2]}|{model}"


@st.cache_data(show_spinner=False, max_entries=5)
def cached_load(file_name: str, data: bytes, sheet: str | None) -> LoadedDataset:
    return load_dataset(file_name, data, sheet)


@st.cache_data(show_spinner=False, max_entries=5)
def cached_profile(file_name: str, data: bytes, sheet: str | None) -> DatasetProfile:
    return profile_dataset(cached_load(file_name, data, sheet).df)


@st.cache_data(show_spinner=False, max_entries=5)
def cached_facts(file_name: str, data: bytes, sheet: str | None) -> list[Fact]:
    ds = cached_load(file_name, data, sheet)
    return build_fact_sheet(ds.df, cached_profile(file_name, data, sheet), ds.cleaning_log)


@st.cache_data(show_spinner=False, max_entries=5)
def cached_sheets(data: bytes) -> list[str]:
    return list_excel_sheets(data)


def formatter(ds: LoadedDataset) -> Formatter:
    return Formatter(ds.currency_symbol)


# ---------------------------------------------------------------- AI settings

def ai_settings() -> tuple[str | None, str]:
    """API key + model from environment (.env), falling back to Streamlit secrets (for cloud hosting)."""
    key, model = get_settings()
    if key is None:
        try:
            key = st.secrets.get("OPENAI_API_KEY") or None
            model = st.secrets.get("OPENAI_MODEL", model)
        except Exception:   # no secrets file at all — that's fine
            pass
    return key, model


def requests_left() -> int:
    return config.MAX_AI_REQUESTS_PER_SESSION - st.session_state.setdefault("ai_requests_used", 0)


def use_request() -> None:
    st.session_state["ai_requests_used"] = st.session_state.get("ai_requests_used", 0) + 1


def usage_caption(model: str) -> str:
    left = max(requests_left(), 0)
    return f"Model {model} · {left} of {config.MAX_AI_REQUESTS_PER_SESSION} AI requests left this session"


def setup_help() -> None:
    st.markdown(
        '<div class="bi-callout info"><div class="title">Connect OpenAI to turn on AI features</div>'
        "Everything else in the app works without it.<ol style='margin:6px 0 0 1.1rem;padding:0'>"
        "<li>Create an API key at <b>platform.openai.com → API keys</b> and add a few dollars of credit.</li>"
        "<li>In the project folder, copy <code>.env.example</code> to a new file named <code>.env</code>.</li>"
        "<li>In <code>.env</code>, set <code>OPENAI_API_KEY=sk-...</code> (your key).</li>"
        "<li>Restart the app (Ctrl+C in Terminal, then <code>streamlit run app.py</code>).</li></ol>"
        "<div style='margin-top:6px'><code>.env</code> is git-ignored, so your key never goes to GitHub.</div></div>",
        unsafe_allow_html=True,
    )
