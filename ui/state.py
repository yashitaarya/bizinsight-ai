"""Cached data steps and shared AI settings used by several tabs.

st.cache_data remembers results for identical inputs, so clicking around the app
doesn't re-read or re-analyze the file. The cache lives in server memory only.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

import streamlit as st

from core import config
from core.data_loader import LoadedDataset, list_excel_sheets, load_dataset
from core.facts import Fact, build_fact_sheet
from core.formatting import Formatter
from core.llm_client import get_settings
from core.profiler import DatasetProfile, profile_dataset

Source = tuple[str, bytes, "str | None"]   # (file name, file bytes, sheet name)

PAGES = ["Home", "Overview", "AI Summary", "Ask AI", "Charts", "Data", "Guide"]


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

def _secret(name: str) -> str | None:
    """A value from Streamlit secrets, or None if there is no secrets file or no such entry."""
    try:
        value = st.secrets.get(name)
    except Exception:   # no secrets file at all — that's fine
        return None
    return str(value) if value not in (None, "") else None


def ai_settings() -> tuple[str | None, str]:
    """API key + model from environment (.env), falling back to Streamlit secrets (for cloud hosting)."""
    key, model = get_settings()
    if key is None:
        key = _secret("OPENAI_API_KEY")
        if key and key.startswith("sk-your-key"):
            key = None   # still the placeholder from secrets.toml.example
        model = _secret("OPENAI_MODEL") or model
    return key, model


# ---------------------------------------------------------------- AI usage limits
# Two limits protect the OpenAI credit of whoever hosts the app:
#   * per browser session (MAX_AI_REQUESTS_PER_SESSION), and
#   * per day across every visitor of this running app (AI_DAILY_LIMIT), because a refresh
#     starts a new session. The shared counter lives in server memory (st.cache_resource),
#     so it resets when the app restarts — a budget limit on the OpenAI key is the final backstop.

def daily_limit() -> int:
    raw = os.environ.get("AI_DAILY_LIMIT") or _secret("AI_DAILY_LIMIT")
    try:
        return max(int(raw), 0) if raw is not None else config.AI_DAILY_LIMIT
    except ValueError:
        return config.AI_DAILY_LIMIT


@st.cache_resource
def _shared_usage() -> dict:
    """One counter shared by every session of this app process."""
    return {"day": None, "count": 0, "lock": threading.Lock()}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def daily_used() -> int:
    usage = _shared_usage()
    with usage["lock"]:
        if usage["day"] != _today():
            usage["day"], usage["count"] = _today(), 0
        return usage["count"]


def session_left() -> int:
    return config.MAX_AI_REQUESTS_PER_SESSION - st.session_state.setdefault("ai_requests_used", 0)


def daily_left() -> int:
    return daily_limit() - daily_used()


def requests_left() -> int:
    return min(session_left(), daily_left())


def use_request() -> None:
    st.session_state["ai_requests_used"] = st.session_state.get("ai_requests_used", 0) + 1
    usage = _shared_usage()
    with usage["lock"]:
        if usage["day"] != _today():
            usage["day"], usage["count"] = _today(), 0
        usage["count"] += 1


def usage_caption(model: str) -> str:
    if daily_left() < session_left():
        left = max(daily_left(), 0)
        return f"Model {model} · {left} of {daily_limit()} AI requests left today (shared by all visitors)"
    left = max(session_left(), 0)
    return f"Model {model} · {left} of {config.MAX_AI_REQUESTS_PER_SESSION} AI requests left this session"


# ---------------------------------------------------------------- shared AI results
# A result generated for a file is reused for anyone who opens the *identical* file (same bytes,
# sheet and model), so the samples cost one request per app restart instead of one per visitor.
# Keys include a SHA-256 hash of the file, so nobody can see results for a file they don't have.

_SHARED_MAX = 200


@st.cache_resource
def _shared_results() -> dict:
    return {"items": {}, "lock": threading.Lock()}


def shared_get(key: str):
    store = _shared_results()
    with store["lock"]:
        return store["items"].get(key)


def shared_put(key: str, value) -> None:
    store = _shared_results()
    with store["lock"]:
        items = store["items"]
        items.pop(key, None)
        items[key] = value
        while len(items) > _SHARED_MAX:      # forget the oldest
            items.pop(next(iter(items)))


def summary_entry(source: Source, model: str):
    """This session's summary for the file, or one generated earlier for the identical file."""
    fingerprint = summary_key(source, model)
    state = st.session_state.setdefault("ai_summaries", {})
    if fingerprint not in state:
        shared = shared_get("summary|" + fingerprint)
        if shared is not None:
            state[fingerprint] = shared
    return state.get(fingerprint)


def limit_message() -> str:
    """Why AI requests are unavailable right now (call when requests_left() <= 0)."""
    if daily_left() <= 0:
        return ("This app has reached its shared daily limit of AI requests, which protects the host's "
                "OpenAI credit. It resets at midnight UTC. Everything else in the app keeps working.")
    return ("This session's limit protects the host's OpenAI credit. "
            "Refresh the page to start a new session.")


def setup_help() -> None:
    st.markdown(
        '<div class="bi-callout info"><div class="title">Connect OpenAI to turn on AI features</div>'
        "Everything else in the app works without it. If you are hosting this app:"
        "<ol style='margin:6px 0 0 1.1rem;padding:0'>"
        "<li>Create an API key at <b>platform.openai.com → API keys</b> and add a few dollars of credit.</li>"
        "<li><b>On your computer:</b> copy <code>.env.example</code> to a new file named <code>.env</code> and set "
        "<code>OPENAI_API_KEY=sk-...</code>, then restart the app.</li>"
        "<li><b>On Streamlit Community Cloud:</b> open the app's <b>Settings → Secrets</b> and add "
        "<code>OPENAI_API_KEY = \"sk-...\"</code>.</li></ol>"
        "<div style='margin-top:6px'>Keys are never stored in the code or on GitHub.</div></div>",
        unsafe_allow_html=True,
    )
