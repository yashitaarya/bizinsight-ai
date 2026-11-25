"""Stage 4 — the "fact sheet": every number the AI is allowed to talk about.

The AI never sees the raw rows. Instead, pandas computes a short list of
numbered facts like

    F7  [trend]  Total Revenue in the last complete month (2025-12) was 244,180.50,
                 that is 23.1% higher than the first complete month (2025-01, 198,355.22).

and the AI must cite these IDs for everything it says. That gives us three things:
  1. Privacy  – only aggregated numbers and column names leave the machine.
  2. Accuracy – all math is done here, in tested Python code.
  3. Checkability – every AI statement can be traced back to a fact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from core.charts import (PARTIAL_PERIOD_THRESHOLD, _RANKABLE_NAME, choose_period, default_agg,
                         period_coverage, value_label)
from core.profiler import DatasetProfile, suggest_kpi_columns

FACT_CATEGORIES = ("overview", "kpi", "trend", "segment", "relationship", "quality")


@dataclass
class Fact:
    id: str
    category: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- formatting
# Facts show full numbers (2,598,123.45) so the AI never has to calculate or guess.

def num(x: float) -> str:
    x = float(x)
    return f"{int(round(x)):,}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"


def pct(x: float) -> str:
    return f"{x:.1f}%"


def change(new: float, old: float) -> str | None:
    """'23.1% higher' / '4.0% lower' — or None when a % change isn't meaningful."""
    if old == 0 or pd.isna(old) or pd.isna(new):
        return None
    diff = (new - old) / abs(old) * 100
    if abs(diff) < 0.05:
        return "about the same as"
    return f"{pct(abs(diff))} {'higher' if diff > 0 else 'lower'} than"


# ---------------------------------------------------------------- individual sections

def _overview(df: pd.DataFrame, profile: DatasetProfile) -> list[tuple[str, str]]:
    roles = pd.Series([c.role for c in profile.columns]).value_counts()
    role_text = ", ".join(f"{n} {r}" for r, n in roles.items())
    facts = [("overview", f"The dataset has {profile.n_rows:,} rows and {profile.n_columns} columns "
                          f"({role_text}).")]
    for d in profile.by_role("date")[:1]:
        if d.stats:
            facts.append(("overview", f"'{d.name}' covers {d.stats['min']} to {d.stats['max']} "
                                      f"({d.stats['span_days']:,} days)."))
    return facts


def _kpis(profile: DatasetProfile) -> list[tuple[str, str]]:
    facts = []
    kpis = suggest_kpi_columns(profile, limit=3)
    for name in kpis:
        s = profile.column(name).stats
        if not s:
            continue
        if default_agg(name) == "sum":
            facts.append(("kpi", f"Total {name} is {num(s['sum'])} (average {num(s['mean'])} per row, "
                                 f"median {num(s['median'])})."))
        else:
            facts.append(("kpi", f"Average {name} is {num(s['mean'])} (median {num(s['median'])}, "
                                 f"range {num(s['min'])} to {num(s['max'])})."))
        if s["mean"] and s["median"] and s["mean"] > 1.5 * s["median"] > 0:
            facts.append(("kpi", f"Average {name} ({num(s['mean'])}) is well above the median "
                                 f"({num(s['median'])}): a minority of large values account for a big part of the total."))
    return facts


_REVENUE = re.compile(r"revenue|sales|income", re.IGNORECASE)
_COST = re.compile(r"cost|expense|cogs", re.IGNORECASE)


# ---------------------------------------------------------------- structured calculations
# These return plain numbers. The fact sentences below AND the KPI cards in the UI
# (core/highlights.py) are both built from them, so they can never disagree.

def margin_values(df: pd.DataFrame, profile: DatasetProfile) -> dict | None:
    """Revenue minus cost over rows where both are known — or None if the data has no such pair."""
    measures = [c.name for c in profile.by_role("measure")]
    revenue = next((m for m in measures if _REVENUE.search(m)), None)
    cost = next((m for m in measures if _COST.search(m)), None)
    if not revenue or not cost:
        return None
    both = df[[revenue, cost]].dropna()
    rev, cst = float(both[revenue].sum()), float(both[cost].sum())
    if rev <= 0:
        return None
    return {"revenue": revenue, "cost": cost, "revenue_total": rev, "cost_total": cst,
            "difference": rev - cst, "margin_pct": (rev - cst) / rev * 100, "left_out": len(df) - len(both)}


def complete_period_series(df: pd.DataFrame, date: str, measure: str | None) -> dict | None:
    """The measure per week/month/quarter, with partly covered first/last periods removed."""
    agg = default_agg(measure)
    frame = df.dropna(subset=[date])
    if frame.empty:
        return None
    code, period_name = choose_period(frame[date])
    if code == "D":
        return None   # day-by-day comparisons are too noisy to summarize
    grouped = frame.groupby(frame[date].dt.to_period(code))
    series = (grouped.size() if measure is None else getattr(grouped[measure], agg)()).sort_index()
    first, last = frame[date].min(), frame[date].max()
    if period_coverage(first.normalize(), series.index[0].end_time.normalize(), series.index[0]) < PARTIAL_PERIOD_THRESHOLD:
        series = series.iloc[1:]
    if len(series) and period_coverage(series.index[-1].start_time, last.normalize(), series.index[-1]) < PARTIAL_PERIOD_THRESHOLD:
        series = series.iloc[:-1]
    if len(series) < 3:
        return None
    return {"series": series, "period": period_name, "label": value_label(measure, agg), "agg": agg,
            "measure": measure, "date": date}


def segment_values(df: pd.DataFrame, profile: DatasetProfile, measure: str | None, limit: int = 4) -> list[dict]:
    """The measure per group for the most useful category columns (product-like ones first)."""
    agg = default_agg(measure)
    cats = [c for c in profile.by_role("category") if 2 <= c.unique <= 50]
    # Product-like columns first; among those, the most varied (Product before Customer Segment).
    cats.sort(key=lambda c: (not _RANKABLE_NAME.search(c.name), -c.unique, profile.columns.index(c)))
    out = []
    for c in cats[:limit]:
        labels = df[c.name].astype(object).where(df[c.name].notna(), "(blank)").astype(str)
        grouped = df.groupby(labels)
        values = (grouped.size() if measure is None else getattr(grouped[measure], agg)()).sort_values(ascending=False)
        if len(values) < 2:
            continue
        total = float(values.sum())
        out.append({"column": c.name, "values": values, "total": total, "label": value_label(measure, agg),
                    "additive": agg in ("sum", "count") and total > 0 and bool((values >= 0).all())})
    return out


# ---------------------------------------------------------------- fact sentences

def _margin(df: pd.DataFrame, profile: DatasetProfile) -> list[tuple[str, str]]:
    """Revenue minus cost — computed here so the AI never has to subtract numbers itself."""
    m = margin_values(df, profile)
    if not m:
        return []
    note = f" ({m['left_out']:,} rows missing either value were left out)" if m["left_out"] else ""
    return [("kpi", f"{m['revenue']} minus {m['cost']} is {num(m['difference'])}, which is "
                    f"{pct(m['margin_pct'])} of {m['revenue']}{note}.")]


def _trend(df: pd.DataFrame, date: str, measure: str | None) -> list[tuple[str, str]]:
    t = complete_period_series(df, date, measure)
    if not t:
        return []
    series, period_name, label, agg = t["series"], t["period"], t["label"], t["agg"]
    name = str
    n = len(series)
    facts = [
        ("trend", f"Across {n} complete {period_name}s ({name(series.index[0])} to {name(series.index[-1])}), "
                  f"{label} per {period_name} ranged from {num(series.min())} ({name(series.idxmin())}) "
                  f"to {num(series.max())} ({name(series.idxmax())}); the average per {period_name} was {num(series.mean())}."),
    ]
    c = change(series.iloc[-1], series.iloc[0])
    if c:
        facts.append(("trend", f"{label} in the last complete {period_name} ({name(series.index[-1])}) was "
                               f"{num(series.iloc[-1])}, {c} the first complete {period_name} "
                               f"({name(series.index[0])}, {num(series.iloc[0])})."))
    if n >= 4:
        half = n // 2
        first_half, second_half = series.iloc[:half], series.iloc[-half:]
        a, b = (first_half.sum(), second_half.sum()) if agg in ("sum", "count") else (first_half.mean(), second_half.mean())
        c = change(b, a)
        if c:
            facts.append(("trend", f"{label} in the last {half} {period_name}s ({name(second_half.index[0])} to "
                                   f"{name(second_half.index[-1])}) was {num(b)}, {c} the first {half} {period_name}s "
                                   f"({name(first_half.index[0])} to {name(first_half.index[-1])}, {num(a)})."))
    return facts


def _segments(df: pd.DataFrame, profile: DatasetProfile, measure: str | None) -> list[tuple[str, str]]:
    facts = []
    for seg in segment_values(df, profile, measure):
        values, total, additive, label, col = seg["values"], seg["total"], seg["additive"], seg["label"], seg["column"]

        def item(k: str) -> str:
            share = f", {pct(values[k] / total * 100)} of the total" if additive else ""
            return f"{k} ({num(values[k])}{share})"

        top_k = min(3, len(values) - 1)
        tops = "; ".join(item(k) for k in values.index[:top_k])
        text = (f"{label} by {col} ({len(values)} groups): highest {top_k} — {tops}. "
                f"Lowest — {item(values.index[-1])}.")
        if additive and len(values) > 4:
            text += (f" The top {top_k} together account for "
                     f"{pct(values.iloc[:top_k].sum() / total * 100)} of the total.")
        facts.append(("segment", text))
    return facts


def _relationships(profile: DatasetProfile) -> list[tuple[str, str]]:
    return [("relationship",
             f"'{c.column_a}' and '{c.column_b}' have a {c.strength} "
             f"{'positive' if c.r > 0 else 'negative'} correlation (r = {c.r:+.2f}, {c.n:,} rows). "
             "Correlation does not show cause and effect.")
            for c in profile.correlations[:3]]


def _quality(profile: DatasetProfile, cleaning_log: list[str]) -> list[tuple[str, str]]:
    facts = [("quality", f"{pct(profile.completeness_pct)} of all cells are filled in; "
                         f"{profile.duplicate_rows:,} rows are exact duplicates.")]
    facts += [("quality", f"[{i.severity}] {i.message}") for i in profile.issues[:8]]
    facts += [("quality", f"While loading, the app made this change: {entry}") for entry in cleaning_log[:5]]
    return facts


# ---------------------------------------------------------------- public API

def build_fact_sheet(df: pd.DataFrame, profile: DatasetProfile, cleaning_log: list[str] | None = None) -> list[Fact]:
    """Compute the numbered facts the AI summary is allowed to use."""
    kpis = suggest_kpi_columns(profile)
    primary = kpis[0] if kpis else None
    dates = [c.name for c in profile.by_role("date") if c.stats.get("distinct_dates", 0) >= 3]

    raw: list[tuple[str, str]] = []
    raw += _overview(df, profile)
    raw += _kpis(profile)
    raw += _margin(df, profile)
    if dates:
        raw += _trend(df, dates[0], primary)
    raw += _segments(df, profile, primary)
    raw += _relationships(profile)
    raw += _quality(profile, cleaning_log or [])

    return [Fact(id=f"F{i}", category=cat, text=text) for i, (cat, text) in enumerate(raw, start=1)]
