"""BizInsight AI — Streamlit entry point.

This file sets up the page, the sidebar and the tabs. Each tab lives in `ui/`,
and all data logic lives in `core/` (which never imports Streamlit).

Run with:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core import config
from core.data_loader import DataLoadError
from ui.components import section_label
from ui.ask_tab import ask_tab
from ui.charts_tab import charts_tab
from ui.home import home_page
from ui.overview import columns_tab, overview_tab
from ui.guide import guide_page
from ui.state import PAGES, Source, cached_load, cached_profile, cached_sheets, formatter
from ui.summary_tab import summary_tab
from ui.theme import CSS

# Read secrets (OPENAI_API_KEY, OPENAI_MODEL) from a local .env file into environment
# variables. .env is git-ignored, so the key never ends up in the code or on GitHub.
# Variables already set in the environment take priority over the file.
load_dotenv(Path(__file__).parent / ".env")

SAMPLES = {
    "Retail sales": Path(__file__).parent / "sample_data" / "sales_sample.csv",
    "Customer support tickets": Path(__file__).parent / "sample_data" / "support_tickets.csv",
}
NO_SAMPLE = "None"

st.set_page_config(page_title="BizInsight AI", page_icon=":material/insights:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def sidebar() -> Source | None:
    """Draw the sidebar. Returns (file name, file bytes, sheet) or None."""
    with st.sidebar:
        st.markdown("### BizInsight AI")
        st.caption("Turn business data into actionable insights. Upload a CSV or Excel file to automatically "
                   "generate interactive dashboards, visualizations, and AI-powered analysis. Explore your data "
                   "through conversational queries, uncover trends and patterns, and get data-backed "
                   "recommendations with transparent calculations behind every result.")
        uploaded = st.file_uploader(
            "Upload a CSV or Excel file", key="upload",
            type=[ext.lstrip(".") for ext in config.ALLOWED_EXTENSIONS],
            help=f"Max {config.MAX_FILE_MB} MB. Files stay in memory and are never saved.",
        )
        sample = st.selectbox("Or explore a sample dataset", [NO_SAMPLE, *SAMPLES], key="sample",
                              disabled=uploaded is not None,
                              help="Synthetic (made-up) data included with the app.")
        st.divider()
        st.caption("Your file is processed in memory for this session only and is never stored. "
                   "AI features send only column names and calculated results — never rows.")

    if uploaded is not None:
        name, data = uploaded.name, uploaded.getvalue()
    elif sample != NO_SAMPLE:
        name, data = SAMPLES[sample].name, SAMPLES[sample].read_bytes()
    else:
        return None

    sheet = None
    if name.lower().endswith(".xlsx"):
        try:
            sheets = cached_sheets(data)
        except DataLoadError as err:
            st.error(f"Couldn't load the file. {err}")
            return None
        sheet = st.sidebar.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
    return name, data, sheet


def _pick_sample(name: str) -> None:
    st.session_state["sample"] = name
    st.session_state["show_guide"] = False


def _show_guide(show: bool) -> None:
    st.session_state["show_guide"] = show


def welcome() -> None:
    st.markdown('<div class="bi-kicker">Business intelligence for spreadsheets</div>'
                '<div class="bi-hero">BizInsight AI</div>'
                '<div class="bi-hero-sub">Turn your business data into insights you can actually use. Simply upload '
                'a CSV or Excel file to explore interactive dashboards, discover trends, and get clear summaries '
                'of your data. You can also ask questions in plain English and receive helpful answers and '
                'business recommendations, all backed by real calculations so you can understand where every '
                'insight comes from.</div>', unsafe_allow_html=True)

    section_label("Start in one click")
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.button("Explore retail sales sample", type="primary", width="stretch", on_click=_pick_sample,
              args=("Retail sales",), key="welcome_sales")
    c2.button("Explore support tickets sample", width="stretch", on_click=_pick_sample,
              args=("Customer support tickets",), key="welcome_tickets")
    c3.button("How to use BizInsight AI", icon=":material/menu_book:", width="stretch", on_click=_show_guide,
              args=(True,), key="welcome_guide")
    st.caption("Or upload your own file in the sidebar (CSV or XLSX, up to 25 MB). Samples are synthetic data.")

    section_label("What you get")
    features = [
        ("Dashboard", "Key metrics chosen for your data, a performance chart and a data-health check."),
        ("Automatic analysis", "Column roles, statistics and data-quality warnings, calculated on upload."),
        ("Interactive charts", "Trends, rankings and comparisons with filters and a chart builder."),
        ("AI summary and Q&A", "Executive briefings and answers that explain calculated results — "
                               "the AI never does the math."),
    ]
    for box, (title, body) in zip(st.columns(4, gap="medium"), features):
        with box.container(border=True):
            st.markdown(f'<div class="bi-nav-title">{title}</div><div class="bi-nav-body">{body}</div>',
                        unsafe_allow_html=True)

    section_label("How it works")
    steps = [("1", "Load", "Your file is read and cleaned in memory; every change is logged."),
             ("2", "Analyze", "pandas profiles the data and calculates numbered facts."),
             ("3", "Explain", "The AI writes summaries and answers from those facts only, and the app checks them.")]
    for box, (n, title, body) in zip(st.columns(3, gap="medium"), steps):
        box.markdown(f'<div class="bi-step">STEP {n}</div><div class="bi-nav-title">{title}</div>'
                     f'<div class="bi-nav-body">{body}</div>', unsafe_allow_html=True)


def data_page(ds, profile, fmt) -> None:
    st.subheader("Data")
    if ds.cleaning_log:
        with st.expander(f"Data preparation ({len(ds.cleaning_log)} automatic changes)"):
            st.caption("Your original file is untouched — these changes apply only to the copy being analyzed.")
            for entry in ds.cleaning_log:
                st.markdown(f"- {entry}")
    preview, columns = st.tabs(["Preview", "Columns"])
    with preview:
        n = st.slider("Rows to show", 5, min(500, len(ds.df)), min(50, len(ds.df))) if len(ds.df) > 5 else len(ds.df)
        st.dataframe(ds.df.head(n), width="stretch", hide_index=True)
    with columns:
        columns_tab(profile, fmt)


def main() -> None:
    source = sidebar()
    if source is None:
        if st.session_state.get("show_guide"):
            st.button("Back to start", icon=":material/arrow_back:", on_click=_show_guide, args=(False,),
                      key="guide_back")
            guide_page()
        else:
            welcome()
        return
    try:
        with st.spinner("Reading and analyzing your file…"):
            ds = cached_load(*source)
            profile = cached_profile(*source)
    except DataLoadError as err:
        st.error(f"Couldn't load the file. {err}")
        return
    fmt = formatter(ds)

    # A new dataset always opens on Home
    loaded = (source[0], len(source[1]), source[2])
    if st.session_state.get("loaded_dataset") != loaded:
        st.session_state["loaded_dataset"] = loaded
        st.session_state["nav"] = "Home"
    st.session_state.setdefault("nav", "Home")
    page = st.segmented_control("Navigation", PAGES, key="nav", required=True, label_visibility="collapsed")
    page = page or "Home"
    if page != "Home":
        st.markdown(f'<div class="bi-kicker" style="margin-top:0.6rem">{ds.file_name}</div>', unsafe_allow_html=True)

    is_sample = st.session_state.get("upload") is None   # no upload -> the data came from a sample
    if page == "Home":
        home_page(source, ds, profile, fmt, is_sample)
    elif page == "Overview":
        st.subheader("Overview")
        overview_tab(profile, fmt)
    elif page == "AI Summary":
        summary_tab(source, ds, profile, fmt)
    elif page == "Ask AI":
        ask_tab(source, ds, profile, fmt)
    elif page == "Charts":
        charts_tab(ds.df, profile, fmt)
    elif page == "Guide":
        guide_page()
    else:
        data_page(ds, profile, fmt)


main()
