"""Reusable display pieces: KPI cards, insight cards, callouts, headlines.

Everything that came from the data or the AI is HTML-escaped before it is placed
inside our HTML, so a column called "<script>" is shown as text, never run.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from core.formatting import tidy_ai_text
from core.highlights import Highlight


def md_safe(text: str) -> str:
    """Escape '$' so Streamlit Markdown never treats '$1,200 … $3,400' as a math formula."""
    return (text or "").replace("$", "\\$")


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section_label(text: str) -> None:
    _html(f'<div class="bi-section">{escape(text)}</div>')


def headline(text: str, sub: str = "", small: bool = False) -> None:
    cls = "bi-headline small" if small else "bi-headline"
    _html(f'<div class="{cls}">{escape(text)}</div>' + (f'<div class="bi-sub">{escape(sub)}</div>' if sub else ""))


ARROWS = {"up": "▲ ", "down": "▼ "}


def kpi_row(items: list[Highlight], compact: bool = False) -> None:
    """A row of KPI cards. The grid has exactly one column per card (wrapping on narrow screens)."""
    if not items:
        return
    cards = "".join(
        f'<div class="bi-kpi"><div class="label" title="{escape(h.label)}">{escape(h.label)}</div>'
        f'<div class="value" title="{escape(h.value)}">{escape(h.value)}</div>'
        + (f'<div class="note {escape(h.trend)}">{ARROWS.get(h.trend, "")}{escape(h.note)}</div>' if h.note else "")
        + "</div>"
        for h in items
    )
    cls = "bi-kpis compact" if compact else "bi-kpis"
    style = f"grid-template-columns: repeat({len(items)}, minmax(0, 1fr));"
    _html(f'<div class="{cls}" style="{style}">{cards}</div>')


def insight_cards(items, numbered: bool = False, unverified_label: str = "Figure not verified") -> None:
    """Cards with a bold title and one line of detail. `items` need .title, .text, .unverified_numbers."""
    for i, item in enumerate(items, start=1):
        num = f'<span class="num">{i}.</span>' if numbered else ""
        tag = ""
        if getattr(item, "unverified_numbers", None):
            tag = (f'<span class="bi-tag" title="Could not match {escape(", ".join(item.unverified_numbers))} '
                   f'to the calculated facts">{escape(unverified_label)}</span>')
        title, text = tidy_ai_text(item.title), tidy_ai_text(item.text)
        detail = f'<div class="detail">{escape(text)}</div>' if text else ""
        _html(f'<div class="bi-card"><div class="title">{num}{escape(title)}{tag}</div>{detail}</div>')


def callout(title: str, lines: list[str], tone: str = "neutral") -> None:
    if not lines:
        return
    body = "".join(f"<li>{escape(line)}</li>" for line in lines)
    _html(f'<div class="bi-callout {tone}"><div class="title">{escape(title)}</div><ul>{body}</ul></div>')


def empty_state(kicker: str, title: str, body: str) -> None:
    _html(f'<div class="bi-empty"><div class="kicker">{escape(kicker)}</div>'
          f'<div class="title">{escape(title)}</div><div class="body">{escape(body)}</div></div>')


def number_columns(table, money_columns: list[str], currency: str | None) -> dict:
    """Streamlit column formats: thousands separators everywhere, currency where it applies."""
    import pandas as pd

    out = {}
    for c in table.columns:
        if pd.api.types.is_numeric_dtype(table[c]):
            if currency and c in money_columns:
                fmt = {"$": "dollar", "€": "euro", "¥": "yen"}.get(currency, "localized")
                out[c] = st.column_config.NumberColumn(format=fmt)
            elif "%" in str(c):
                out[c] = st.column_config.NumberColumn(format="%.1f%%")
            else:
                out[c] = st.column_config.NumberColumn(format="localized")
    return out


def key_charts(df, profile, key_prefix: str, height: int = 300) -> None:
    """The main trend and the top-performer chart for this dataset (if its columns allow them)."""
    from core.charts import make_chart, recommend_charts

    specs = recommend_charts(profile, max_charts=10)
    chosen = [s for s in specs if s.kind == "time_series"][:1] + \
             [s for s in specs if s.kind in ("ranking", "category_bar")][:1]
    if not chosen:
        st.caption("This dataset has no date or category columns to chart.")
        return
    widths = [3, 2] if len(chosen) == 2 else [1]
    for box, spec in zip(st.columns(widths, gap="medium"), chosen):
        with box.container(border=True):
            st.markdown(f"**{spec.title}**")
            try:
                result = make_chart(spec, df)
                result.figure.update_layout(height=height)
                st.plotly_chart(result.figure, width="stretch", theme=None, key=f"{key_prefix}_{spec.kind}",
                                config={"displaylogo": False})
                for note in result.notes:
                    st.caption(note)
            except ValueError as err:
                st.caption(str(err))


def dataset_header(name: str, chips: list[str], status: str, status_note: str) -> None:
    chip_html = "".join(f'<span class="bi-chip">{escape(c)}</span>' for c in chips)
    _html(f'<div class="bi-dataset"><div><div class="bi-kicker">Dataset</div>'
          f'<div class="bi-dataset-name">{escape(name)}</div><div class="bi-chips">{chip_html}</div></div>'
          f'<div class="bi-status"><span class="bi-dot"></span>{escape(status)}'
          f'<div class="bi-status-note">{escape(status_note)}</div></div></div>')


def panel_title(title: str, subtitle: str = "") -> None:
    _html(f'<div class="bi-panel-title">{escape(title)}</div>'
          + (f'<div class="bi-panel-sub">{escape(subtitle)}</div>' if subtitle else ""))
