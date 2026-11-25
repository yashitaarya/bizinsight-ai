"""Stage 6 — headline figures ("KPI cards") shown above AI summaries and answers.

These are computed here, in Python, from the same calculations as the fact sheet.
The AI never chooses or writes these numbers; it only writes the words around them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.facts import change, complete_period_series, margin_values, segment_values
from core.formatting import Formatter, period_label
from core.profiler import DatasetProfile, suggest_kpi_columns


@dataclass
class Highlight:
    label: str          # short label, e.g. "Total Revenue"
    value: str          # formatted, e.g. "$2.60M" or "Laptop"
    note: str = ""      # one short line of context, e.g. "45.0% of total"
    trend: str = ""     # "up" / "down" / "" — only colors the note, never changes the value


def pct_change(new: float, old: float) -> float | None:
    if old in (0, None) or pd.isna(old) or pd.isna(new):
        return None
    return (new - old) / abs(old) * 100


def summary_highlights(df: pd.DataFrame, profile: DatasetProfile, fmt: Formatter, limit: int = 4) -> list[Highlight]:
    """Up to four headline figures for the dataset as a whole. Adapts to whatever columns exist."""
    kpis = suggest_kpi_columns(profile)
    primary = kpis[0] if kpis else None
    out: list[Highlight] = []

    if primary:
        s = profile.column(primary).stats
        if s:
            from core.charts import default_agg
            if default_agg(primary) == "sum":
                out.append(Highlight(f"Total {primary}", fmt.number(s["sum"], primary),
                                     f"Average {fmt.number(s['mean'], primary)} per row"))
            else:
                out.append(Highlight(f"Average {primary}", fmt.number(s["mean"], primary),
                                     f"Median {fmt.number(s['median'], primary)}"))
    else:
        out.append(Highlight("Rows", fmt.count(profile.n_rows), f"{profile.n_columns} columns"))

    dates = [c.name for c in profile.by_role("date") if c.stats.get("distinct_dates", 0) >= 3]
    if dates:
        t = complete_period_series(df, dates[0], primary)
        if t:
            series, period = t["series"], t["period"]
            delta = pct_change(series.iloc[-1], series.iloc[0])
            note = f"Latest complete {period}"
            trend = ""
            if delta is not None and change(series.iloc[-1], series.iloc[0]) != "about the same as":
                note = f"{delta:+.1f}% vs {period_label(series.index[0])}"
                trend = "up" if delta > 0 else "down"
            out.append(Highlight(f"{t['label']}, {period_label(series.index[-1])}",
                                 fmt.number(series.iloc[-1], primary), note, trend))

    m = margin_values(df, profile)
    if m:
        out.append(Highlight(f"{m['revenue']} minus {m['cost']}", fmt.number(m["difference"], m["revenue"]),
                             f"{fmt.percent(m['margin_pct'])} of {m['revenue']}"))

    segments = segment_values(df, profile, primary, limit=1)
    if segments:
        seg = segments[0]
        top, value = seg["values"].index[0], seg["values"].iloc[0]
        if seg["additive"]:
            note = f"{fmt.number(value, primary)} · {fmt.percent(value / seg['total'] * 100)} of total"
        else:
            note = f"{seg['label']} {fmt.number(value, primary)}"
        out.append(Highlight(f"Highest {seg['column']}", str(top), note))

    if len(out) < limit:
        out.append(Highlight("Data completeness", fmt.percent(profile.completeness_pct),
                             f"{profile.duplicate_rows:,} duplicate rows"))
    return out[:limit]


# ---------------------------------------------------------------- homepage dashboard metrics
# The homepage picks its KPI cards from the column roles found by the profiler, so the
# same code produces "Orders / Revenue / Avg revenue per order" for sales data and
# "Tickets / Avg resolution hours / Reopened mix" for support data — nothing is hard-coded.

import re as _re

_ID_SUFFIX = _re.compile(r"[\s_\-]*(id|number|no|code|#)$", _re.IGNORECASE)
_STATUS_TIERS = [
    _re.compile(r"status|state|stage", _re.IGNORECASE),
    _re.compile(r"outcome|result|resolved|reopened|closed|churn|returned|cancel", _re.IGNORECASE),
    _re.compile(r"priority|severity|tier|type|segment", _re.IGNORECASE),
]


def record_noun(profile: DatasetProfile) -> str:
    """What one row represents, from the first ID column: 'Order ID' -> 'Orders'. Falls back to 'Records'."""
    for c in profile.by_role("identifier"):
        stem = _ID_SUFFIX.sub("", c.name).strip()
        if stem and stem.lower() != c.name.lower() and stem.isascii() and stem.replace(" ", "").isalpha():
            stem = stem.title()
            return stem if stem.endswith("s") else stem + "s"
    return "Records"


def status_column(profile: DatasetProfile):
    """A small category column that describes the state or type of each record, if any."""
    # 2-6 values that actually repeat (a "mix" where every value appears once says nothing)
    candidates = [c for c in profile.by_role("category") + profile.by_role("boolean")
                  if 2 <= c.unique <= 6 and c.unique < c.count]
    for tier in _STATUS_TIERS:
        for c in candidates:
            if tier.search(c.name):
                return c
    return min(candidates, key=lambda c: c.unique) if candidates else None


def _mix_highlight(col, fmt: Formatter) -> Highlight:
    """E.g. "Priority mix: Medium — 40% of rows · Low 35% · High 18%", from the profiler's value counts."""
    top = col.stats["top_values"]
    first = top[0]
    rest = " · ".join(f"{v['value']} {fmt.percent(v['pct'], 0)}" for v in top[1:3])
    note = f"{fmt.percent(first['pct'], 0)} of rows" + (f" · {rest}" if rest else "")
    return Highlight(f"{col.name} mix", str(first["value"]), note)


def dashboard_metrics(df: pd.DataFrame, profile: DatasetProfile, fmt: Formatter, limit: int = 5) -> list[Highlight]:
    """The homepage KPI cards, chosen from the columns this dataset actually has."""
    from core.charts import default_agg

    noun = record_noun(profile)
    ids = profile.by_role("identifier")
    n_records = ids[0].unique if ids and noun != "Records" else profile.n_rows
    date_cols = [c for c in profile.by_role("date") if c.stats]
    if n_records != profile.n_rows:
        first_note = f"{profile.n_rows:,} rows"
    elif date_cols:
        first_note = f"{date_cols[0].stats['min'][:10]} to {date_cols[0].stats['max'][:10]}"
    else:
        first_note = ""
    cards = [Highlight(noun, fmt.count(n_records), first_note)]

    kpis = suggest_kpi_columns(profile, limit=4)
    additive = [k for k in kpis if default_agg(k) == "sum"]
    averaged = [k for k in kpis if default_agg(k) != "sum"]
    dates = [c.name for c in profile.by_role("date") if c.stats.get("distinct_dates", 0) >= 3]

    def trend_note(measure: str | None) -> tuple[str, str]:
        if not dates:
            return "", ""
        t = complete_period_series(df, dates[0], measure)
        if not t:
            return "", ""
        delta = pct_change(t["series"].iloc[-1], t["series"].iloc[0])
        if delta is None or change(t["series"].iloc[-1], t["series"].iloc[0]) == "about the same as":
            return "", ""
        return (f"{delta:+.1f}% {period_label(t['series'].index[-1])} vs {period_label(t['series'].index[0])}",
                "up" if delta > 0 else "down")

    if additive:
        main = additive[0]
        s = profile.column(main).stats
        note, trend = trend_note(main)
        cards.append(Highlight(f"Total {main}", fmt.number(s["sum"], main), note or f"Median {fmt.number(s['median'], main)} per row", trend))
        per = s["sum"] / n_records if n_records else None
        if per is not None:
            unit = noun[:-1].lower() if noun != "Records" else "record"
            cards.append(Highlight(f"Avg {main} per {unit}", fmt.number(per, main),
                                   f"Across {fmt.count(n_records)} {noun.lower()}"))
        m = margin_values(df, profile)
        if m:
            cards.append(Highlight(f"{m['revenue']} minus {m['cost']}", fmt.number(m["difference"], m["revenue"]),
                                   f"{fmt.percent(m['margin_pct'])} of {m['revenue']} (estimate)"))

    for measure in averaged:
        if len(cards) >= limit - 1:
            break
        c = profile.column(measure)
        note, trend = trend_note(measure) if measure == (averaged[0] if not additive else None) else ("", "")
        missing = f" · {c.missing_pct:.0f}% missing" if c.missing_pct >= 5 else ""
        cards.append(Highlight(f"Avg {measure}", fmt.number(c.stats["mean"], measure),
                               note or f"Median {fmt.number(c.stats['median'], measure)}{missing}", trend))

    if not additive and dates and len(cards) < limit - 1:
        # Volume over time, e.g. "Tickets per week" — counts of rows per complete period
        t = complete_period_series(df, dates[0], None)
        if t:
            cards.append(Highlight(f"{noun} per {t['period']}", fmt.number(t["series"].mean()),
                                   f"Average of {len(t['series'])} complete {t['period']}s"))

    status = status_column(profile)
    if status is not None and status.stats and len(cards) < limit:
        cards.append(_mix_highlight(status, fmt))
    return cards[:limit]
