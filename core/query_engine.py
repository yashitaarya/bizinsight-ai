"""Stage 5 — the safe query engine behind natural-language questions.

The AI never writes code. It fills in a *plan*: a small form with fixed fields
(which operation, which column, which filters). This module:

  1. validate_plan()  – checks every field against the real dataset: columns
                        must exist and have the right role, filter values must
                        actually occur, numbers must be sensible. Anything else
                        is rejected with a plain-English reason.
  2. run_query()      – executes the validated plan with ordinary pandas code
                        and returns a table, a chart spec and numbered result
                        facts (Q1, Q2, …) for the AI to explain.

Only these operations exist — the "approved menu":
    aggregate       one number: total / average / median / min / max / row count
    group_by        a measure compared across the groups of a category
    time_trend      a measure per day / week / month / quarter / year
    correlation     how strongly two measures move together
    count_distinct  how many different values a column has
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.charts import (AGG_LABELS, PARTIAL_PERIOD_THRESHOLD, PERIOD_NAMES, ChartSpec, _labels,
                         apply_filters, choose_period, period_coverage)
from core.facts import Fact, change, num, pct
from core.formatting import Formatter, period_label
from core.highlights import Highlight, pct_change
from core.profiler import DatasetProfile

OPERATIONS = ("aggregate", "group_by", "time_trend", "correlation", "count_distinct")
AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")
PERIODS = {"auto": None, "day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
NUMERIC_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "=": np.equal}
MAX_GROUPS = 50
MAX_PERIOD_FACTS = 24


class QueryError(Exception):
    """The plan can't be run on this dataset. The message is safe to show to users."""


@dataclass
class QueryPlan:
    operation: str
    metric: str | None = None             # measure column; None = count rows
    subtract_metric: str | None = None    # optional: use (metric − subtract_metric), e.g. Revenue − Cost
    aggregation: str = "sum"
    group_by: str | None = None           # category column (or, for count_distinct, any column)
    date_column: str | None = None
    period: str = "auto"
    second_metric: str | None = None      # for correlation
    sort: str = "desc"
    limit: int = 10
    filters: dict[str, list[str]] = field(default_factory=dict)            # category -> allowed values
    date_from: pd.Timestamp | None = None
    date_to: pd.Timestamp | None = None
    numeric_filters: list[tuple[str, str, float]] = field(default_factory=list)  # (column, op, value)


@dataclass
class QueryResult:
    title: str
    table: pd.DataFrame
    facts: list[Fact]
    rows_used: int
    filter_text: str = ""
    chart: ChartSpec | None = None
    frame: pd.DataFrame | None = None     # the filtered rows the chart should use (stays local, never sent)
    notes: list[str] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)   # headline figures for the UI


# ---------------------------------------------------------------- validation

def _role(profile: DatasetProfile, column: str) -> str:
    return next(c.role for c in profile.columns if c.name == column)


def _column(profile: DatasetProfile, name, allowed_roles: tuple[str, ...] | None, what: str) -> str | None:
    if name in (None, ""):
        return None
    names = [c.name for c in profile.columns]
    if name not in names:
        # Be forgiving about letter case, but nothing else
        matches = [n for n in names if n.lower() == str(name).lower()]
        if not matches:
            raise QueryError(f"There's no column called '{name}' in this dataset.")
        name = matches[0]
    if allowed_roles and _role(profile, name) not in allowed_roles:
        role = _role(profile, name)
        article = "an" if role[0] in "aeiou" else "a"
        raise QueryError(f"'{name}' can't be used as {what} (it's {article} {role} column).")
    return name


def _date(value, label: str) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as err:
        raise QueryError(f"'{value}' isn't a valid {label} date.") from err


def validate_plan(raw: dict, df: pd.DataFrame, profile: DatasetProfile) -> QueryPlan:
    """Turn the AI's plan (a plain dict) into a checked QueryPlan, or raise QueryError."""
    op = raw.get("operation")
    if op not in OPERATIONS:
        raise QueryError(f"'{op}' isn't one of the supported analyses.")
    agg = raw.get("aggregation") or "sum"
    if agg not in AGGREGATIONS:
        raise QueryError(f"'{agg}' isn't a supported calculation.")
    period = raw.get("period") or "auto"
    if period not in PERIODS:
        raise QueryError(f"'{period}' isn't a supported time period.")

    metric = _column(profile, raw.get("metric"), ("measure",), "a number to calculate with")
    subtract = _column(profile, raw.get("subtract_metric"), ("measure",), "a number to subtract")
    if subtract and not metric:
        raise QueryError("A subtraction needs a main number column too.")
    if agg == "count":
        metric, subtract = None, None
    elif metric is None and op in ("aggregate", "group_by", "time_trend"):
        agg = "count"   # nothing to add up -> count rows

    plan = QueryPlan(
        operation=op, metric=metric, subtract_metric=subtract, aggregation=agg, period=period,
        sort="asc" if raw.get("sort") == "asc" else "desc",
        limit=int(min(max(raw.get("limit") or 10, 1), MAX_GROUPS)),
    )

    if op == "group_by":
        plan.group_by = _column(profile, raw.get("group_by"), ("category", "boolean"), "a grouping")
        if not plan.group_by:
            raise QueryError("Comparing groups needs a category column such as Region or Product.")
    elif op == "count_distinct":
        plan.group_by = _column(profile, raw.get("group_by") or raw.get("metric"), None, "a column")
        if not plan.group_by:
            raise QueryError("Counting distinct values needs a column.")
    elif op == "correlation":
        plan.metric = _column(profile, raw.get("metric"), ("measure",), "a number")
        plan.second_metric = _column(profile, raw.get("second_metric"), ("measure",), "a number")
        if not plan.metric or not plan.second_metric or plan.metric == plan.second_metric:
            raise QueryError("A relationship question needs two different numeric columns.")
        plan.subtract_metric, plan.aggregation = None, "count"

    date_needed = op == "time_trend" or raw.get("date_from") or raw.get("date_to")
    if date_needed:
        dates = [c.name for c in profile.by_role("date")]
        plan.date_column = _column(profile, raw.get("date_column") or (dates[0] if dates else None),
                                   ("date",), "a date")
        if not plan.date_column:
            raise QueryError("This dataset has no date column, so time-based questions can't be answered.")
    plan.date_from, plan.date_to = _date(raw.get("date_from"), "start"), _date(raw.get("date_to"), "end")
    if plan.date_from and plan.date_to and plan.date_from > plan.date_to:
        raise QueryError("The start date is after the end date.")

    for f in raw.get("filters") or []:
        col = _column(profile, f.get("column"), ("category", "boolean"), "a filter")
        if not col:
            continue
        available = {v.lower(): v for v in _labels(df[col]).unique()}
        wanted = [str(v) for v in f.get("values") or []]
        unknown = [v for v in wanted if v.lower() not in available]
        if unknown:
            examples = ", ".join(sorted(available.values())[:8])
            raise QueryError(f"'{col}' has no value called {', '.join(repr(u) for u in unknown)}. "
                             f"Available values include: {examples}.")
        if wanted:
            plan.filters[col] = [available[v.lower()] for v in wanted]

    for f in raw.get("numeric_filters") or []:
        col = _column(profile, f.get("column"), ("measure",), "a number filter")
        op_ = f.get("op")
        if op_ not in NUMERIC_OPS:
            raise QueryError(f"'{op_}' isn't a supported comparison.")
        try:
            value = float(f.get("value"))
        except (TypeError, ValueError) as err:
            raise QueryError(f"'{f.get('value')}' isn't a number.") from err
        plan.numeric_filters.append((col, op_, value))
    return plan


# ---------------------------------------------------------------- execution helpers

def describe_filters(plan: QueryPlan) -> str:
    parts = [f"{col} is {' or '.join(vals)}" for col, vals in plan.filters.items()]
    if plan.date_from or plan.date_to:
        start = plan.date_from.strftime("%Y-%m-%d") if plan.date_from else "the start"
        end = plan.date_to.strftime("%Y-%m-%d") if plan.date_to else "the end"
        parts.append(f"{plan.date_column} from {start} to {end}")
    parts += [f"{col} {op} {num(v)}" for col, op, v in plan.numeric_filters]
    return "; ".join(parts)


def _filtered(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    out = apply_filters(df, plan.date_column, plan.date_from, plan.date_to, plan.filters)
    for col, op, value in plan.numeric_filters:
        out = out[NUMERIC_OPS[op](out[col], value).fillna(False).astype(bool)]
    return out


def _metric_series(frame: pd.DataFrame, plan: QueryPlan) -> tuple[pd.DataFrame, str | None, str]:
    """Add the (possibly derived) metric column. Returns (frame, column to use, its display name)."""
    if plan.metric is None:
        return frame, None, "rows"
    if plan.subtract_metric:
        name = f"{plan.metric} minus {plan.subtract_metric}"
        frame = frame.assign(**{name: frame[plan.metric] - frame[plan.subtract_metric]})
        return frame, name, name
    return frame, plan.metric, plan.metric


def _label(metric_name: str | None, agg: str) -> str:
    return "Number of rows" if metric_name is None or agg == "count" else f"{AGG_LABELS[agg]} {metric_name}"


def _agg(values, agg: str):
    return values.size() if agg == "count" else getattr(values, agg)()


def _scope(filter_text: str) -> str:
    return f" (where {filter_text})" if filter_text else ""


# ---------------------------------------------------------------- operations

def _run_aggregate(frame, plan, col, label, scope, fmt):
    value = len(frame) if plan.aggregation == "count" else getattr(frame[col].dropna(), plan.aggregation)()
    if plan.aggregation != "count" and pd.isna(value):
        raise QueryError("There are no values to calculate with after filtering.")
    used = len(frame) if col is None else int(frame[col].notna().sum())
    facts = [f"{label}{scope} is {num(value)}, based on {used:,} rows."]
    if col and used < len(frame):
        facts.append(f"{len(frame) - used:,} matching rows had no value and were left out.")
    table = pd.DataFrame({"Result": [label], "Value": [value], "Rows used": [used]})
    hl = [Highlight(label, fmt.number(value, col) if plan.aggregation != "count" else fmt.count(value),
                    f"Based on {used:,} rows")]
    return table, facts, None, hl


def _run_group_by(frame, plan, col, label, scope, fmt):
    labels = _labels(frame[plan.group_by])
    grouped = frame.assign(**{plan.group_by: labels}).groupby(plan.group_by)
    values = (grouped.size() if col is None else _agg(grouped[col], plan.aggregation)).dropna()
    if values.empty:
        raise QueryError("There are no values to compare after filtering.")
    values = values.sort_values(ascending=plan.sort == "asc")
    n_groups, total = len(values), values.sum()
    show_share = plan.aggregation in ("sum", "count") and total > 0 and (values >= 0).all()
    shown = values.head(plan.limit)

    facts = []
    for rank, (group, v) in enumerate(shown.items(), start=1):
        share = f", {pct(v / total * 100)} of the total" if show_share else ""
        facts.append(f"#{rank} {'lowest' if plan.sort == 'asc' else 'highest'}: {plan.group_by} '{group}' — "
                     f"{label} {num(v)}{share}{scope}.")
    summary = f"There are {n_groups} {plan.group_by} groups{scope}."
    if show_share:
        summary += f" Their combined {label.lower()} is {num(total)}."
    if n_groups > len(shown):
        summary += f" Only {len(shown)} are listed."
    facts.append(summary)
    if n_groups >= 2:
        first, second, last = values.iloc[0], values.iloc[1], values.iloc[-1]
        word = "lower" if plan.sort == "asc" else "higher"
        gap = abs(first - second)
        facts.append(f"'{values.index[0]}' is {num(gap)} {word} than '{values.index[1]}' "
                     f"({change(first, second) or 'no % change'} it).")
        if n_groups > 2:
            facts.append(f"The gap between '{values.index[0]}' and '{values.index[-1]}' is {num(abs(first - last))}.")

    table = shown.rename(label).reset_index()
    if show_share:
        table["Share of total %"] = (shown / total * 100).round(1).values
    # The chart shows at least 5 groups for context, even when the question asked only for "the top one".
    spec = ChartSpec("ranking", f"{label} by {plan.group_by}", "", x=plan.group_by, y=col,
                     agg=plan.aggregation, top_n=max(plan.limit, min(n_groups, 5)), ascending=plan.sort == "asc")

    def value_text(v):
        if plan.aggregation == "count":
            text = f"{fmt.count(v)} rows"
        elif show_share:
            text = fmt.number(v, col)
        else:
            text = f"{AGG_LABELS[plan.aggregation]} {fmt.number(v, col)}"
        return text + (f" · {fmt.percent(v / total * 100)} of total" if show_share else "")

    word = "Lowest" if plan.sort == "asc" else "Highest"
    hl = [Highlight(f"{word} {plan.group_by}", str(values.index[0]), value_text(values.iloc[0]))]
    if n_groups >= 2:
        hl.append(Highlight(f"Runner-up {plan.group_by}", str(values.index[1]), value_text(values.iloc[1])))
    hl.append(Highlight(f"{plan.group_by} groups", fmt.count(n_groups),
                        f"Combined {fmt.count(total) if plan.aggregation == 'count' else fmt.number(total, col)}"
                        if show_share else "Compared in this result"))
    if spec.top_n > len(shown):
        hl_note = f"The chart shows the top {spec.top_n} {plan.group_by} values for comparison."
        if plan.sort == "asc":
            hl_note = hl_note.replace("top", "lowest")
        spec.reason = hl_note
    return table, facts, spec, hl


def _run_time_trend(frame, plan, col, label, scope, fmt):
    frame = frame.dropna(subset=[plan.date_column])
    if frame.empty:
        raise QueryError(f"No rows have a value in '{plan.date_column}' after filtering.")
    code = PERIODS[plan.period] or choose_period(frame[plan.date_column])[0]
    name = PERIOD_NAMES[code]
    periods = frame[plan.date_column].dt.to_period(code)
    grouped = frame.groupby(periods)
    series = (grouped.size() if col is None else _agg(grouped[col], plan.aggregation)).sort_index()

    first, last = frame[plan.date_column].min(), frame[plan.date_column].max()
    partial = set()
    if code != "D":
        if period_coverage(first.normalize(), series.index[0].end_time.normalize(), series.index[0]) < PARTIAL_PERIOD_THRESHOLD:
            partial.add(series.index[0])
        if period_coverage(series.index[-1].start_time, last.normalize(), series.index[-1]) < PARTIAL_PERIOD_THRESHOLD:
            partial.add(series.index[-1])

    facts = []
    listed = series if len(series) <= MAX_PERIOD_FACTS else series.iloc[-12:]
    if len(listed) < len(series):
        facts.append(f"There are {len(series)} {name}s; only the last 12 are listed individually.")
    for p, v in listed.items():
        flag = f" (only partly covered by the data)" if p in partial else ""
        facts.append(f"{name.title()} {p}: {label} {num(v)}{flag}{scope}.")

    complete = series[[p not in partial for p in series.index]]
    if len(complete) >= 2:
        facts.append(f"Among {len(complete)} complete {name}s, the highest {label.lower()} was {num(complete.max())} "
                     f"({complete.idxmax()}) and the lowest was {num(complete.min())} ({complete.idxmin()}).")
        c = change(complete.iloc[-1], complete.iloc[0])
        if c:
            facts.append(f"The last complete {name} ({complete.index[-1]}, {num(complete.iloc[-1])}) was {c} "
                         f"the first complete {name} ({complete.index[0]}, {num(complete.iloc[0])}).")
    table = pd.DataFrame({name.title(): [period_label(p) for p in series.index], label: series.values,
                          "Complete?": ["no" if p in partial else "yes" for p in series.index]})
    spec = ChartSpec("time_series", f"{label} per {name}", "", x=plan.date_column, y=col,
                     agg=plan.aggregation, period=code)

    def v(x):
        return fmt.count(x) if plan.aggregation == "count" else fmt.number(x, col)

    hl = []
    basis = complete if len(complete) else series
    hl.append(Highlight(f"Latest complete {name}", v(basis.iloc[-1]), period_label(basis.index[-1])))
    if len(complete) >= 2:
        delta = pct_change(complete.iloc[-1], complete.iloc[0])
        if delta is not None:
            hl.append(Highlight("Change", f"{delta:+.1f}%", f"{period_label(complete.index[0])} to "
                                f"{period_label(complete.index[-1])}", "up" if delta > 0 else "down"))
        hl.append(Highlight(f"Highest {name}", period_label(complete.idxmax()), v(complete.max())))
    return table, facts, spec, hl


def _run_correlation(frame, plan, scope):
    pair = frame[[plan.metric, plan.second_metric]].dropna()
    if len(pair) < 3 or pair[plan.metric].nunique() < 2 or pair[plan.second_metric].nunique() < 2:
        raise QueryError("There aren't enough varied values to measure a relationship.")
    r = float(pair[plan.metric].corr(pair[plan.second_metric]))
    if abs(r) < 0.2:
        verdict = "no meaningful correlation"
    else:
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        verdict = f"a {strength} {'positive' if r > 0 else 'negative'} correlation"
    facts = [f"'{plan.metric}' and '{plan.second_metric}' have {verdict} "
             f"(r = {r:+.2f}, {len(pair):,} rows){scope}.",
             "r runs from -1 to +1; correlation shows a pattern, not cause and effect."]
    table = pd.DataFrame({"Column A": [plan.metric], "Column B": [plan.second_metric],
                          "Correlation r": [round(r, 3)], "Rows used": [len(pair)]})
    spec = ChartSpec("scatter", f"{plan.second_metric} vs {plan.metric}", "", x=plan.metric, y=plan.second_metric)
    hl = [Highlight("Correlation (r)", f"{r:+.2f}", verdict[0].upper() + verdict[1:]),
          Highlight("Rows compared", f"{len(pair):,}", "Rows with both values")]
    return table, facts, spec, hl


def _run_count_distinct(frame, plan, scope):
    s = frame[plan.group_by].dropna()
    n = int(s.nunique())
    facts = [f"'{plan.group_by}' has {n:,} different values{scope}, across {len(s):,} rows with a value."]
    table = pd.DataFrame({"Column": [plan.group_by], "Distinct values": [n], "Rows with a value": [len(s)]})
    return table, facts, None, [Highlight(f"Distinct {plan.group_by} values", f"{n:,}", f"Across {len(s):,} rows")]


# ---------------------------------------------------------------- public API

def run_query(plan: QueryPlan, df: pd.DataFrame, fmt: Formatter | None = None) -> QueryResult:
    """Execute a validated plan with pandas. Raises QueryError if nothing can be computed."""
    fmt = fmt or Formatter()
    frame = _filtered(df, plan)
    filter_text = describe_filters(plan)
    scope = _scope(filter_text)
    if frame.empty:
        raise QueryError(f"No rows match{scope or ' the question'}, so there's nothing to calculate.")

    frame, col, metric_name = _metric_series(frame, plan)
    label = _label(None if col is None else metric_name, plan.aggregation)

    if plan.operation == "aggregate":
        table, texts, spec, hl = _run_aggregate(frame, plan, col, label, scope, fmt)
        title = label
    elif plan.operation == "group_by":
        table, texts, spec, hl = _run_group_by(frame, plan, col, label, scope, fmt)
        title = f"{label} by {plan.group_by}"
    elif plan.operation == "time_trend":
        table, texts, spec, hl = _run_time_trend(frame, plan, col, label, scope, fmt)
        title = spec.title
    elif plan.operation == "correlation":
        table, texts, spec, hl = _run_correlation(frame, plan, scope)
        title = f"Relationship between {plan.metric} and {plan.second_metric}"
    else:
        table, texts, spec, hl = _run_count_distinct(frame, plan, scope)
        title = f"Distinct values of {plan.group_by}"

    notes = []
    if len(frame) < len(df):
        notes.append(f"Calculated on {len(frame):,} of {len(df):,} rows{scope}.")
    facts = [Fact(f"Q{i}", "result", t) for i, t in enumerate(texts, start=1)]
    return QueryResult(title, table, facts, len(frame), filter_text, spec, frame, notes, hl)
