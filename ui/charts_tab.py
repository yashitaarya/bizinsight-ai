"""Charts tab: filters, recommended charts and a chart builder (no AI involved)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.charts import AGG_LABELS, ChartSpec, apply_filters, default_agg, make_chart, recommend_charts, value_label
from core.formatting import Formatter
from core.profiler import ColumnProfile, DatasetProfile, suggest_kpi_columns
from ui.components import number_columns, section_label


def _filterable_categories(profile: DatasetProfile) -> list[ColumnProfile]:
    return [c for c in profile.by_role("category") + profile.by_role("boolean") if 2 <= c.unique <= 50]


def filter_bar(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """Date range + up to 3 category filters. Returns the filtered rows."""
    dates = [c for c in profile.by_role("date") if c.stats]
    cats = _filterable_categories(profile)[:3]
    if not dates and not cats:
        return df

    date_col, start, end, chosen = None, None, None, {}
    with st.container(border=True):
        st.markdown("**Filters** · apply to every chart on this tab")
        boxes = st.columns((1 if dates else 0) + len(cats))
        if dates:
            date_col = dates[0].name
            lo, hi = df[date_col].min().date(), df[date_col].max().date()
            picked = boxes[0].date_input(f"{date_col} range", value=(lo, hi), min_value=lo, max_value=hi,
                                         key="filter_dates")
            if isinstance(picked, (tuple, list)) and len(picked) == 2:   # ignore half-picked ranges
                start, end = picked
        for box, c in zip(boxes[1 if dates else 0:], cats):
            options = sorted(df[c.name].dropna().astype(str).unique())
            chosen[c.name] = box.multiselect(c.name, options, placeholder="All", key=f"filter_{c.name}")

    filtered = apply_filters(df, date_col, start, end, chosen)
    if len(filtered) < len(df):
        st.caption(f"Showing **{len(filtered):,}** of {len(df):,} rows.")
    return filtered


def show_chart(spec: ChartSpec, df: pd.DataFrame, full_df: pd.DataFrame, key: str, fmt: Formatter,
               describe: bool = True) -> None:
    """Draw one chart with its title, reason, notes and the numbers behind it."""
    # Colors are fixed by each category's size in the *unfiltered* data, so filters never repaint them.
    order = full_df[spec.color].astype(str).value_counts().index.tolist() if spec.color else None
    st.markdown(f"**{spec.title}**")
    if describe and spec.reason:
        st.caption(spec.reason)
    try:
        result = make_chart(spec, df, order)
    except ValueError as err:
        st.caption(str(err))
        return
    st.plotly_chart(result.figure, width="stretch", theme=None, key=key, config={"displaylogo": False})
    for note in result.notes:
        st.caption(note)
    with st.expander("Show the numbers"):
        st.dataframe(result.table, hide_index=True, width="stretch",
                     column_config=number_columns(result.table, fmt.money_columns(result.table.columns), fmt.currency))


def chart_builder(df: pd.DataFrame, full_df: pd.DataFrame, profile: DatasetProfile, fmt: Formatter) -> None:
    measures = [c.name for c in profile.by_role("measure")]
    dates = [c.name for c in profile.by_role("date")]
    cats = [c.name for c in profile.by_role("category") + profile.by_role("boolean") if c.unique >= 2]

    # Only offer chart types this dataset can support — and say why others are missing.
    types: dict[str, str] = {}
    unavailable: list[str] = []
    for label, kind, ok, need in [
        ("Trend over time", "time", bool(dates), "a date column"),
        ("Compare categories", "category", bool(cats), "a category column (e.g. Region)"),
        ("Spread of values", "distribution", bool(measures), "a numeric column"),
        ("Two numbers together", "scatter", len(measures) >= 2, "two numeric columns"),
    ]:
        if ok:
            types[label] = kind
        else:
            unavailable.append(f"{label} needs {need}")
    if not types:
        st.caption("This dataset has no dates, categories or numbers to chart.")
        return
    if unavailable:
        st.caption("Not available for this data: " + "; ".join(unavailable) + ".")

    kind = types[st.radio("Chart type", list(types), horizontal=True, key="builder_type")]
    agg_names = {"Total": "sum", "Average": "mean", "Median": "median"}
    rows_label = "(number of rows)"

    def metric_pickers(box_metric, box_agg) -> tuple[str | None, str]:
        kpis = suggest_kpi_columns(profile)
        start = 1 + measures.index(kpis[0]) if kpis else 0   # default to the main KPI, e.g. Revenue
        y = box_metric.selectbox("Measure", [rows_label] + measures, index=start, key=f"builder_y_{kind}")
        if y == rows_label:
            return None, "count"
        names = list(agg_names)
        default = names.index(AGG_LABELS[default_agg(y)])
        return y, agg_names[box_agg.selectbox("Calculate", names, index=default, key=f"builder_agg_{kind}",
                                              help="Total adds values up; Average/Median suit prices, rates, scores.")]

    c1, c2, c3, c4 = st.columns(4)
    if kind == "time":
        x = c1.selectbox("Date column", dates, key="builder_date")
        y, agg = metric_pickers(c2, c3)
        split = c4.selectbox("Split by", ["(none)"] + [c for c in cats if profile.column(c).unique <= 50],
                             key="builder_split")
        color = None if split == "(none)" else split
        spec = ChartSpec("time_by_category" if color else "time_series",
                         f"{value_label(y, agg)} over time" + (f" by {color}" if color else ""),
                         reason="", x=x, y=y, agg=agg, color=color)
    elif kind == "category":
        x = c1.selectbox("Group by", cats, key="builder_cat")
        y, agg = metric_pickers(c2, c3)
        top_n = c4.slider("Show top", 3, 30, 10, key="builder_top")
        spec = ChartSpec("ranking", f"{value_label(y, agg)} by {x}", reason="", x=x, y=y, agg=agg, top_n=top_n)
    elif kind == "distribution":
        x = c1.selectbox("Numeric column", measures, key="builder_dist")
        spec = ChartSpec("distribution", f"How '{x}' values are spread", reason="", x=x)
    else:
        x = c1.selectbox("Horizontal axis", measures, key="builder_sx")
        y = c2.selectbox("Vertical axis", [m for m in measures if m != x], key="builder_sy")
        spec = ChartSpec("scatter", f"{y} vs {x}", reason="Each dot is one row.", x=x, y=y)

    with st.container(border=True):
        show_chart(spec, df, full_df, key="builder_chart", fmt=fmt)


def charts_tab(full_df: pd.DataFrame, profile: DatasetProfile, fmt: Formatter) -> None:
    df = filter_bar(full_df, profile)
    if df.empty:
        st.warning("No rows match these filters. Try selecting more values or a wider date range.")
        return

    st.subheader("Recommended charts")
    specs = recommend_charts(profile)
    if not specs:
        st.caption("No columns here make a meaningful chart. Charts need a date column, "
                   "a category column with 2–50 different values, or a numeric column.")
    else:
        st.caption("Picked automatically from your column types. Hover over any chart for exact values.")
        for row_start in range(0, len(specs), 2):
            for box, (i, spec) in zip(st.columns(2, gap="medium"),
                                      enumerate(specs[row_start:row_start + 2], start=row_start)):
                with box.container(border=True):
                    show_chart(spec, df, full_df, key=f"recommended_{i}", fmt=fmt)

    section_label("Build your own chart")
    chart_builder(df, full_df, profile, fmt)
