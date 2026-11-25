"""Stage 3 — chart recommendations and Plotly figures.

How it fits together:

    DatasetProfile ──► recommend_charts() ──► [ChartSpec, ChartSpec, …]
                                                   │
    DataFrame (optionally filtered) ──► make_chart(spec, df) ──► ChartResult
                                                                 (figure, table, notes)

* A `ChartSpec` is a small *description* of a chart ("Revenue by Region, summed").
  Keeping it separate from the figure makes recommendations easy to test, and in
  Stage 5 the AI will be able to request charts in this same format.
* All numbers are computed with pandas here, then handed to Plotly to draw.
* Every chart comes with the table of numbers behind it and plain-English `notes`
  about anything the reader should know (e.g. "only the top 10 of 25 shown").
* We only recommend a chart when the columns it needs actually exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.profiler import DatasetProfile, is_additive, suggest_kpi_columns

# ---------------------------------------------------------------- style
# A color-blind-checked palette, always used in this fixed order so that a given
# category keeps the same color everywhere (and when filters change).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#a3a29c"
TEXT, MUTED, GRID, AXIS = "#0f172a", "#52514e", "#ebeae6", "#d4d3cd"

KINDS = ("time_series", "time_by_category", "category_bar", "ranking", "distribution", "scatter")
AGG_LABELS = {"sum": "Total", "mean": "Average", "median": "Median", "min": "Minimum", "max": "Maximum",
              "count": "Number of rows"}

PERIOD_NAMES = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
MAX_LINES = 5          # more than this many lines on one chart becomes unreadable
MAX_SCATTER_POINTS = 5_000

_RANKABLE_NAME = re.compile(
    r"product|item|sku|customer|client|brand|store|branch|employee|rep\b|salesperson|vendor|"
    r"supplier|city|country|account|campaign|channel|course|model",
    re.IGNORECASE,
)


@dataclass
class ChartSpec:
    kind: str                 # one of KINDS
    title: str
    reason: str               # why this chart was recommended (shown to the user)
    x: str | None = None      # date or category column (or the measure, for distribution/scatter)
    y: str | None = None      # measure column; None means "count rows"
    agg: str = "sum"          # sum | mean | median | count
    color: str | None = None  # category column used to split lines
    top_n: int = 10
    period: str | None = None # time bucket ("D", "W", "M", "Q", "Y"); None = choose automatically
    ascending: bool = False   # rankings: show the lowest values first (e.g. "fastest response")


@dataclass
class ChartResult:
    figure: go.Figure
    table: pd.DataFrame                       # the numbers behind the chart
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- small helpers

def default_agg(measure: str | None) -> str:
    """Sum things that add up (Revenue); average things that don't (Price, Rating)."""
    if measure is None:
        return "count"
    return "sum" if is_additive(measure) else "mean"


def value_label(y: str | None, agg: str) -> str:
    if y is None or agg == "count":
        return "Number of rows"
    return f"{AGG_LABELS[agg]} {y}"


def choose_period(dates: pd.Series) -> tuple[str, str]:
    """Pick a time bucket so a trend has a readable number of points.

    Returns a pandas period code and a friendly name, e.g. ("M", "month").
    """
    dates = dates.dropna()
    span = (dates.max() - dates.min()).days if not dates.empty else 0
    if span <= 45:
        return "D", "day"
    if span <= 200:
        return "W", "week"
    if span <= 3 * 366:
        return "M", "month"
    return "Q", "quarter"


def _labels(s: pd.Series) -> pd.Series:
    """Category values as text, with missing values shown honestly as '(blank)'."""
    return s.astype(object).where(s.notna(), "(blank)").astype(str)


def _aggregate(frame: pd.DataFrame, keys: list[str], y: str | None, agg: str) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=False, observed=True)
    values = grouped.size() if (y is None or agg == "count") else getattr(grouped[y], agg)()
    return values.reset_index(name="value")


def _missing_note(df: pd.DataFrame, y: str | None, agg: str) -> list[str]:
    if y is None or agg == "count":
        return []
    missing = int(df[y].isna().sum())
    return [f"{missing:,} row(s) with no '{y}' value were left out."] if missing else []


def color_map(order: list[str] | None) -> dict[str, str]:
    """Fixed color per category value (by its rank in the *full* dataset), so
    filtering never repaints the categories that remain."""
    mapping = {str(v): PALETTE[i] for i, v in enumerate((order or [])[: len(PALETTE)])}
    mapping["Other"] = OTHER_COLOR
    mapping["(blank)"] = OTHER_COLOR
    return mapping


def _value_range(values: pd.Series) -> list[float] | None:
    """Bars start at zero, with ~18% spare room so the value labels aren't cut off."""
    if values.empty:
        return None
    lo, hi = min(0.0, float(values.min())), max(0.0, float(values.max()))
    pad = (hi - lo) * 0.18 or 1.0
    return [lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)]


def _style(fig: go.Figure, x_title: str, y_title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=16, t=36, b=8),
        font=dict(family="Source Sans Pro, Helvetica, Arial, sans-serif", size=13, color=TEXT),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        bargap=0.35,
        barcornerradius=4,
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=AXIS, tickfont_color=MUTED)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont_color=MUTED)
    return fig


# ---------------------------------------------------------------- filters

def apply_filters(
    df: pd.DataFrame,
    date_col: str | None = None,
    start=None,
    end=None,
    category_filters: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Keep rows inside the date range (inclusive) and matching the chosen categories.

    An empty list for a category means "no filter" (keep everything).
    """
    mask = pd.Series(True, index=df.index)
    if date_col:
        if start is not None:
            mask &= df[date_col] >= pd.Timestamp(start)
        if end is not None:
            mask &= df[date_col] < pd.Timestamp(end) + pd.Timedelta(days=1)  # include the whole end day
    for col, values in (category_filters or {}).items():
        if values:
            mask &= _labels(df[col]).isin([str(v) for v in values])
    return df[mask]


# ---------------------------------------------------------------- chart builders

PARTIAL_PERIOD_THRESHOLD = 0.9   # flag a week/month/quarter if data covers less than 90% of its days


def period_coverage(start: pd.Timestamp, end: pd.Timestamp, period: pd.Period) -> float:
    """Share of a period's days that fall between start and end (inclusive)."""
    period_days = (period.end_time.normalize() - period.start_time).days + 1
    return ((end - start).days + 1) / period_days


def _time_chart(spec: ChartSpec, df: pd.DataFrame, order: list[str] | None) -> ChartResult:
    date, y, agg = spec.x, spec.y, spec.agg
    frame = df.dropna(subset=[date]).copy()
    notes = _missing_note(frame, y, agg)
    if frame.empty:
        raise ValueError(f"No rows have a value in '{date}'.")

    code, name = (spec.period, PERIOD_NAMES[spec.period]) if spec.period else choose_period(frame[date])
    frame["period"] = frame[date].dt.to_period(code).dt.start_time

    keys = ["period"]
    if spec.kind == "time_by_category" and spec.color:
        labels = _labels(frame[spec.color])
        totals = _aggregate(frame.assign(_c=labels), ["_c"], y, agg).sort_values("value", ascending=False)
        keep = totals["_c"].head(MAX_LINES).tolist()
        frame[spec.color] = labels.where(labels.isin(keep), "Other")
        if len(totals) > MAX_LINES:
            notes.append(f"Showing the {MAX_LINES} largest {spec.color} values; the rest are grouped as 'Other'.")
        keys.append(spec.color)

    data = _aggregate(frame, keys, y, agg).sort_values(keys)

    # Is the first/last bucket mostly missing from the data? Then its total will look artificially low.
    if code != "D":
        first, last = frame[date].min(), frame[date].max()
        last_p, first_p = last.to_period(code), first.to_period(code)
        if period_coverage(last_p.start_time, last.normalize(), last_p) < PARTIAL_PERIOD_THRESHOLD:
            effect = ("so its total may look lower than it really is" if agg in ("sum", "count")
                      else "so its value is based on fewer rows than the others")
            notes.append(f"The last {name} ({last_p.start_time:%Y-%m-%d} onward) is only partly covered "
                         f"— the data ends on {last:%Y-%m-%d}, {effect}.")
        if period_coverage(first.normalize(), first_p.end_time.normalize(), first_p) < PARTIAL_PERIOD_THRESHOLD:
            notes.append(f"The first {name} is only partly covered — the data starts on {first:%Y-%m-%d}.")

    ylab = value_label(y, agg)
    many = len(data) > 40
    if spec.color in data.columns:
        fig = px.line(data, x="period", y="value", color=spec.color, markers=not many,
                      color_discrete_map=color_map(order))
    else:
        fig = px.line(data, x="period", y="value", markers=not many, color_discrete_sequence=PALETTE)
    fig.update_traces(line_width=2, marker_size=8,
                      hovertemplate=f"%{{x|%Y-%m-%d}}<br>{ylab}: %{{y:,.2f}}<extra>%{{fullData.name}}</extra>")
    fig.update_layout(hovermode="x unified")
    if code in ("Q", "Y"):
        # Label quarters/years as "Q1 2025" / "2025" instead of month names
        starts = sorted(data["period"].unique())
        text = [f"Q{(pd.Timestamp(s).month - 1) // 3 + 1} {pd.Timestamp(s).year}" if code == "Q"
                else str(pd.Timestamp(s).year) for s in starts]
        fig.update_xaxes(tickvals=starts, ticktext=text)
    if agg in ("sum", "count") and (data["value"] >= 0).all():
        # Start totals at zero, otherwise small dips look dramatic
        fig.update_yaxes(rangemode="tozero")
    table = data.rename(columns={"period": f"{name.title()} starting", "value": ylab})
    return ChartResult(_style(fig, f"{date} (by {name})", ylab), table, notes)


def _category_chart(spec: ChartSpec, df: pd.DataFrame) -> ChartResult:
    cat, y, agg = spec.x, spec.y, spec.agg
    notes = _missing_note(df, y, agg)
    frame = df.assign(**{cat: _labels(df[cat])})
    data = _aggregate(frame, [cat], y, agg).sort_values("value", ascending=spec.ascending)
    total_groups = len(data)
    if total_groups > spec.top_n:
        which = "lowest" if spec.ascending else "top"
        notes.append(f"Showing the {which} {spec.top_n} of {total_groups:,} {cat} values.")
        data = data.head(spec.top_n)

    ylab = value_label(y, agg)
    longest = data[cat].str.len().max() if not data.empty else 0
    horizontal = spec.kind == "ranking" or len(data) > 8 or longest > 12
    if horizontal:
        # Largest bar at the top
        fig = px.bar(data, x="value", y=cat, orientation="h", color_discrete_sequence=PALETTE[:1])
        fig.update_yaxes(categoryorder="array", categoryarray=data[cat].tolist()[::-1])
        fig.update_traces(texttemplate="%{x:,.3~s}", textposition="outside", cliponaxis=False,
                          hovertemplate=f"%{{y}}<br>{ylab}: %{{x:,.2f}}<extra></extra>")
        fig = _style(fig, ylab, "", height=max(300, 34 * len(data) + 90))
        fig.update_xaxes(showgrid=True, gridcolor=GRID, range=_value_range(data["value"]))
        fig.update_yaxes(showgrid=False)
    else:
        fig = px.bar(data, x=cat, y="value", color_discrete_sequence=PALETTE[:1])
        fig.update_traces(texttemplate="%{y:,.3~s}", textposition="outside", cliponaxis=False,
                          hovertemplate=f"%{{x}}<br>{ylab}: %{{y:,.2f}}<extra></extra>")
        fig = _style(fig, cat, ylab)
        fig.update_yaxes(range=_value_range(data["value"]))
    fig.update_traces(textfont_color=MUTED)
    return ChartResult(fig, data.rename(columns={"value": ylab}).reset_index(drop=True), notes)


def _distribution_chart(spec: ChartSpec, df: pd.DataFrame) -> ChartResult:
    col = spec.x
    values = df[col].dropna().astype(float)
    if values.empty:
        raise ValueError(f"'{col}' has no values to chart.")
    counts, edges = np.histogram(values, bins=min(30, max(5, values.nunique())))
    centres = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(
        x=centres, y=counts, width=(edges[1:] - edges[:-1]) * 0.94, marker_color=PALETTE[0],
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        hovertemplate="%{customdata[0]:,.2f} to %{customdata[1]:,.2f}<br>Rows: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=float(values.median()), line_dash="dot", line_color=MUTED,
                  annotation_text=f"median {values.median():,.2f}", annotation_font_color=MUTED)
    table = pd.DataFrame({"From": edges[:-1].round(2), "To": edges[1:].round(2), "Rows": counts})
    return ChartResult(_style(fig, col, "Number of rows"), table, _missing_note(df, col, "sum"))


def _scatter_chart(spec: ChartSpec, df: pd.DataFrame) -> ChartResult:
    x, y = spec.x, spec.y
    data = df[[x, y]].dropna()
    notes = []
    dropped = len(df) - len(data)
    if dropped:
        notes.append(f"{dropped:,} row(s) missing '{x}' or '{y}' were left out.")
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
        notes.append(f"Showing a random sample of {MAX_SCATTER_POINTS:,} rows to keep the chart fast.")
    fig = px.scatter(data, x=x, y=y, opacity=0.55, color_discrete_sequence=PALETTE[:1])
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")),
                      hovertemplate=f"{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>")
    fig = _style(fig, x, y)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    if len(data) >= 3:
        r = data[x].corr(data[y])
        notes.append(f"Correlation r = {r:+.2f}. A pattern here does not prove that one causes the other.")
    return ChartResult(fig, data.head(200).reset_index(drop=True), notes)


def make_chart(spec: ChartSpec, df: pd.DataFrame, color_order: list[str] | None = None) -> ChartResult:
    """Turn a ChartSpec into a Plotly figure plus the numbers behind it.

    Raises ValueError (with a friendly message) if the data can't support the chart.
    """
    if df.empty:
        raise ValueError("No rows to chart — try loosening the filters.")
    for col in (spec.x, spec.y, spec.color):
        if col is not None and col not in df.columns:
            raise ValueError(f"Column '{col}' isn't in this dataset.")
    if spec.kind in ("time_series", "time_by_category"):
        return _time_chart(spec, df, color_order)
    if spec.kind in ("category_bar", "ranking"):
        return _category_chart(spec, df)
    if spec.kind == "distribution":
        return _distribution_chart(spec, df)
    if spec.kind == "scatter":
        return _scatter_chart(spec, df)
    raise ValueError(f"Unknown chart type '{spec.kind}'.")


# ---------------------------------------------------------------- recommendations

def recommend_charts(profile: DatasetProfile, max_charts: int = 6) -> list[ChartSpec]:
    """Suggest charts that make sense for this dataset, based on column roles.

    Rules (each only applies if the needed columns exist):
      date + measure        → trend over time
      'product-like' column → ranking of the top performers
      small category        → comparison bar chart
      date + small category → one line per category
      measure               → distribution (how values are spread)
      correlated measures   → scatter plot
    With no measures at all, charts count rows instead of summing values.
    """
    kpis = suggest_kpi_columns(profile)
    primary = kpis[0] if kpis else None
    agg = default_agg(primary)
    ylab = value_label(primary, agg)
    what = f"'{primary}'" if primary else "the number of rows"

    dates = [c for c in profile.by_role("date") if c.stats.get("distinct_dates", 0) >= 3]
    cats = [c for c in profile.by_role("category") if 2 <= c.unique <= 50]
    rank_cats = [c for c in cats if c.unique >= 5 and (_RANKABLE_NAME.search(c.name) or c.unique > 12)]
    compare_cats = [c for c in cats if c.unique <= 12 and c not in rank_cats]

    specs: list[ChartSpec] = []

    if dates:
        d = dates[0].name
        specs.append(ChartSpec(
            "time_series", f"{ylab} over time", x=d, y=primary, agg=agg,
            reason=f"'{d}' is a date column, so we can show how {what} changes over time.",
        ))

    for c in rank_cats[:1]:
        title = f"Top {min(10, c.unique)} {c.name} by {ylab}" if c.unique > 10 else f"{c.name} ranked by {ylab}"
        specs.append(ChartSpec(
            "ranking", title, x=c.name, y=primary, agg=agg,
            reason=f"'{c.name}' has {c.unique} different values — ranking them shows the top performers.",
        ))

    if compare_cats:
        c = compare_cats[0]
        specs.append(ChartSpec(
            "category_bar", f"{ylab} by {c.name}", x=c.name, y=primary, agg=agg,
            reason=f"'{c.name}' splits the data into {c.unique} groups — compare {what} across them.",
        ))

    split = next((c for c in compare_cats if c.unique <= MAX_LINES), None)
    if dates and split:
        specs.append(ChartSpec(
            "time_by_category", f"{ylab} over time by {split.name}", x=dates[0].name, y=primary, agg=agg,
            color=split.name,
            reason=f"One line per {split.name} shows whether the groups follow the same trend.",
        ))

    for c in compare_cats[1:2]:
        specs.append(ChartSpec(
            "category_bar", f"{ylab} by {c.name}", x=c.name, y=primary, agg=agg,
            reason=f"Another way to split {what}: by '{c.name}' ({c.unique} groups).",
        ))

    if primary:
        specs.append(ChartSpec(
            "distribution", f"How '{primary}' values are spread", x=primary, agg="count",
            reason=f"Shows whether most '{primary}' values are small, large or in between, "
                   "and how many are unusually high or low.",
        ))

    if profile.correlations:
        c = profile.correlations[0]
        specs.append(ChartSpec(
            "scatter", f"{c.column_b} vs {c.column_a}", x=c.column_a, y=c.column_b, agg="count",
            reason=f"These two columns have a {c.strength} relationship (r = {c.r:+.2f}).",
        ))

    return specs[:max_charts]
