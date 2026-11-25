"""Consistent number formatting for the whole app.

One place decides how numbers look, so every card, table and chart agrees:
    money       $2.60M · $842,552 · $893.93   (symbol only if the data contained one)
    numbers     2.60M · 12,124 · 10.10
    percents    32.7%
    counts      1,200

Which columns are "money"? Measures whose name suggests an amount (Revenue, Cost,
Price, Salary…). The currency symbol comes from the data itself (see
LoadedDataset.currency_symbol) — if the file never showed one, we don't invent it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_MONEY_NAME = re.compile(
    r"revenue|sales|cost|price|amount|profit|income|expense|spend|salary|budget|fee|payment|value|cogs",
    re.IGNORECASE,
)
_PERCENT_NAME = re.compile(r"percent|pct|rate|discount|margin %|%", re.IGNORECASE)


def looks_like_money(column: str | None) -> bool:
    return bool(column) and bool(_MONEY_NAME.search(column)) and not _PERCENT_NAME.search(column)


def compact(x: float) -> str:
    """2,602,869 -> 2.60M; 842,551.6 -> 842,552; 893.93 -> 893.93; 12 -> 12."""
    x = float(x)
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if a >= 1e4 or (a >= 1e3 and x.is_integer()):
        return f"{x:,.0f}"
    if x.is_integer():
        return f"{int(x):,}"
    return f"{x:,.2f}"


@dataclass
class Formatter:
    """Formats values for one dataset (knows its currency symbol, if any)."""

    currency: str | None = None

    def number(self, x, column: str | None = None) -> str:
        if x is None or (not isinstance(x, str) and pd.isna(x)):
            return "—"
        if isinstance(x, str):
            return x
        text = compact(x)
        if self.currency and looks_like_money(column):
            sign = "-" if float(x) < 0 else ""
            return f"{sign}{self.currency}{text.lstrip('-')}"
        return text

    @staticmethod
    def percent(x, digits: int = 1) -> str:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x):.{digits}f}%"

    @staticmethod
    def count(x) -> str:
        return "—" if x is None or pd.isna(x) else f"{int(x):,}"

    def money_columns(self, columns) -> list[str]:
        return [c for c in columns if looks_like_money(str(c))] if self.currency else []


def period_label(p) -> str:
    """Friendly name for a pandas Period: 'Jan 2025', 'Q1 2025', 'Week of 19 May 2025', '2025', '3 Mar 2025'."""
    freq = getattr(p, "freqstr", "") or ""
    start = p.start_time
    if freq.startswith("W"):
        return f"Week of {start.day} {start:%b %Y}"
    if freq.startswith("M"):
        return f"{start:%b %Y}"
    if freq.startswith("Q"):
        return f"Q{p.quarter} {p.year}"
    if freq.startswith(("Y", "A")):
        return f"{p.year}"
    return f"{start.day} {start:%b %Y}"


# ---------------------------------------------------------------- tidying AI-written text
# Applied only when displaying the model's wording, AFTER citations and numbers have been
# checked against the facts — so verification always runs on what the model actually wrote.

import re as _re

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_ID = r"[FQ]\d{1,3}"
_ID_GROUP = _re.compile(rf"\s*\((?:[A-Za-z ]{{0,30}}\s)?({_ID}(?:\s*(?:,|;|&|and)\s*{_ID})*)\)")
_QUARTER = _re.compile(r"\b(\d{4})Q([1-4])\b")
_YEAR_MONTH = _re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b(?!-\d)")
_LONG_NUMBER = _re.compile(r"(?<![\d.])(?<!\d,)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{5,}(?:\.\d+)?)(?!\d)(?!,\d)")


def tidy_ai_text(text: str, known_ids: set[str] | None = None) -> str:
    """Make model-written text match the app's style.

    * removes fact references such as "(F1, F2)" or "(see F22)" — they live in Sources & calculations;
      Q-references are only removed when they are known result IDs (so "(Q4)" as a quarter survives);
    * writes periods as "Q1 2025" / "Dec 2025" instead of "2025Q1" / "2025-12";
    * compacts long numbers: 2,602,869.24 -> 2.60M, 160,818.93 -> 160,819 (values are not changed,
      only rounded for display).
    """
    if not text:
        return text

    def drop_ids(m: _re.Match) -> str:
        ids = _re.findall(_ID, m.group(1))
        removable = all(i.startswith("F") or (known_ids and i in known_ids) for i in ids)
        return "" if removable else m.group(0)

    text = _ID_GROUP.sub(drop_ids, text)
    text = _QUARTER.sub(lambda m: f"Q{m.group(2)} {m.group(1)}", text)
    text = _YEAR_MONTH.sub(lambda m: f"{_MONTHS[int(m.group(2)) - 1]} {m.group(1)}", text)

    def shorten(m: _re.Match) -> str:
        value = float(m.group(1).replace(",", ""))
        return compact(value) if abs(value) >= 10_000 else m.group(1)

    text = _LONG_NUMBER.sub(shorten, text)
    return _re.sub(r"\s+([.,;])", r"\1", text).strip()
