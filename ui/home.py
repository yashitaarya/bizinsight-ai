"""Home: a dashboard-style landing page for the loaded dataset.

  dataset header (name, status, file type, size, date range)
  key metrics      — chosen from column roles (core/highlights.dashboard_metrics)
  performance      — the main trend + top-performer charts (shared with AI Summary)
  AI highlights    — only from a summary the user generated; otherwise a prompt to generate one
  data health      — completeness, duplicates and the most important quality issues
  explore          — shortcuts to every other page
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.data_loader import LoadedDataset
from core.formatting import Formatter, tidy_ai_text
from core.highlights import Highlight, dashboard_metrics
from core.profiler import DatasetProfile
from ui.components import (callout, md_safe, dataset_header, empty_state, insight_cards, key_charts, kpi_row, panel_title,
                           section_label)
from ui.state import Source, ai_settings, go_to, requests_left, summary_key

NAV_CARDS = [
    ("Overview", "Overview", "Key figures, column roles and data quality."),
    ("Charts", "Charts", "Recommended charts, filters and a chart builder."),
    ("AI Summary", "AI Summary", "Executive briefing and recommendations."),
    ("Ask AI", "Ask AI", "Plain-English questions answered from your data."),
    ("Data", "Data & columns", "Browse rows and inspect every column."),
]


def _header(source: Source, ds: LoadedDataset, profile: DatasetProfile, is_sample: bool) -> None:
    file_type = Path(ds.file_name).suffix.lstrip(".").upper() or "File"
    chips = [file_type + (f" · sheet '{ds.sheet_name}'" if ds.sheet_name else ""),
             f"{profile.n_rows:,} records", f"{profile.n_columns} columns"]
    dates = [d for d in profile.by_role("date") if d.stats]
    if dates:
        chips.append(f"{dates[0].stats['min'][:10]} to {dates[0].stats['max'][:10]}")
    if ds.currency_symbol:
        chips.append(f"Amounts in {ds.currency_symbol}")
    chips.append(f"{profile.completeness_pct:.1f}% complete")
    status = "Sample dataset loaded" if is_sample else "File uploaded"
    changes = len(ds.cleaning_log)
    note = f"{changes} automatic preparation step{'s' if changes != 1 else ''}" if changes else "No changes needed"
    dataset_header(ds.file_name, chips, status, note)


def _ai_highlights(source: Source) -> None:
    key, model = ai_settings()
    entry = st.session_state.get("ai_summaries", {}).get(summary_key(source, model))
    with st.container(border=True):
        panel_title("AI highlights", "From your executive summary — every point is linked to a calculation.")
        if entry:
            summary = entry[0]
            if summary.headline:
                st.markdown(f"**{md_safe(tidy_ai_text(summary.headline))}**")
            insight_cards(summary.key_findings[:3])
            st.button("Open full AI summary", key="home_open_summary", on_click=go_to, args=("AI Summary",),
                      type="primary")
        elif key is None:
            empty_state("Not connected", "AI features need an OpenAI API key",
                        "Add your key to the .env file to generate highlights. Everything else works without it.")
            st.button("See setup steps", key="home_setup", on_click=go_to, args=("AI Summary",))
        else:
            empty_state("No summary yet", "Generate an AI summary to see highlights here",
                        "Highlights only appear after a summary has been generated and checked against the "
                        "calculated facts — nothing is shown until then.")
            st.button("Generate AI summary", key="home_generate", type="primary", on_click=go_to,
                      args=("AI Summary", True), disabled=requests_left() <= 0)


def _data_health(profile: DatasetProfile) -> None:
    warnings = [i.message for i in profile.issues if i.severity == "warning"]
    missing = sorted((c for c in profile.columns if c.missing), key=lambda c: -c.missing_pct)
    unusual = [c.name for c in profile.columns if c.stats.get("outliers")]
    with st.container(border=True):
        panel_title("Data health", "Checked automatically when the file was loaded.")
        kpi_row([Highlight("Complete", f"{profile.completeness_pct:.1f}%"),
                 Highlight("Duplicates", f"{profile.duplicate_rows:,}"),
                 Highlight("Warnings", f"{len(warnings)}")], compact=True)
        if warnings:
            callout("Worth fixing", warnings[:2], "warn")
        summary = []
        if missing:
            shown = ", ".join(f"{c.name} {c.missing_pct:.1f}%" for c in missing[:3])
            summary.append(f"Missing values: {shown}" + (f" and {len(missing) - 3} more" if len(missing) > 3 else ""))
        if unusual:
            summary.append(f"Unusually high or low values in: {', '.join(unusual[:4])}"
                           + (f" and {len(unusual) - 4} more" if len(unusual) > 4 else ""))
        if summary:
            callout("Worth knowing", summary, "neutral")
        elif not warnings:
            callout("Looks clean", ["No missing values, duplicates or unusual values detected."], "info")
        st.button("View full data-quality check", key="home_quality", on_click=go_to, args=("Overview",))


def _explore() -> None:
    section_label("Explore")
    for box, (page, title, body) in zip(st.columns(len(NAV_CARDS)), NAV_CARDS):
        with box.container(border=True):
            st.markdown(f'<div class="bi-nav-title">{title}</div><div class="bi-nav-body">{body}</div>',
                        unsafe_allow_html=True)
            st.button("Open", key=f"home_nav_{page}", on_click=go_to, args=(page,), width="stretch")


def home_page(source: Source, ds: LoadedDataset, profile: DatasetProfile, fmt: Formatter, is_sample: bool) -> None:
    _header(source, ds, profile, is_sample)

    section_label("Key metrics")
    kpi_row(dashboard_metrics(ds.df, profile, fmt))

    section_label("Performance overview")
    key_charts(ds.df, profile, key_prefix="home_chart", height=290)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        _ai_highlights(source)
    with right:
        _data_health(profile)

    _explore()
