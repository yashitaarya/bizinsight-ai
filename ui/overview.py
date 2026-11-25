"""Overview and Columns tabs: automatic profiling results (no AI involved)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.charts import default_agg
from core.formatting import Formatter
from core.profiler import ColumnProfile, DatasetProfile, suggest_kpi_columns
from ui.components import callout, section_label

ROLE_INFO = {
    "measure": ("Measure", "Numbers you can add up or average, like Revenue or Units."),
    "category": ("Category", "A small set of repeating labels to group by, like Region."),
    "date": ("Date", "Dates or times — useful for trends."),
    "identifier": ("Identifier", "Unique labels such as Order ID. Not used for math."),
    "boolean": ("Yes/No", "True/false values."),
    "text": ("Text", "Free text with many different values."),
}


# ---------------------------------------------------------------- overview

def dataset_metrics(profile: DatasetProfile) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{profile.n_rows:,}")
    m2.metric("Columns", f"{profile.n_columns:,}")
    m3.metric("Completeness", f"{profile.completeness_pct:.1f}%",
              help=f"Share of cells that aren't empty ({profile.missing_cells:,} empty of {profile.total_cells:,}).")
    m4.metric("Duplicate rows", f"{profile.duplicate_rows:,}")


def key_figures(profile: DatasetProfile, fmt: Formatter) -> None:
    section_label("Key figures")
    dates = [d for d in profile.by_role("date") if d.stats]
    if dates:
        d = dates[0]
        st.caption(f"Data covers {d.stats['min']} to {d.stats['max']} "
                   f"({d.stats['span_days']:,} days, based on '{d.name}').")

    measures = [c.name for c in profile.by_role("measure")]
    if not measures:
        st.caption("No numeric columns were found, so there are no totals to show.")
        return
    chosen = st.multiselect(
        "Columns to summarize", measures, default=suggest_kpi_columns(profile), max_selections=4,
        help="Columns that add up (Revenue, Units) show a total; others (prices, scores, times) show an average.",
    )
    for box, name in zip(st.columns(max(len(chosen), 1)), chosen):
        s = profile.column(name).stats
        if default_agg(name) == "sum":
            box.metric(f"Total {name}", fmt.number(s.get("sum"), name))
            box.caption(f"Average {fmt.number(s.get('mean'), name)} · Median {fmt.number(s.get('median'), name)}")
        else:
            box.metric(f"Average {name}", fmt.number(s.get("mean"), name))
            box.caption(f"Median {fmt.number(s.get('median'), name)} · "
                        f"Range {fmt.number(s.get('min'), name)} – {fmt.number(s.get('max'), name)}")


def column_roles(profile: DatasetProfile) -> None:
    section_label("What's in this dataset")
    counts = pd.Series([c.role for c in profile.columns]).value_counts()
    for box, (role, n) in zip(st.columns(len(counts)), counts.items()):
        name, desc = ROLE_INFO[role]
        names = ", ".join(c.name for c in profile.by_role(role))
        box.metric(name, n, help=f"{desc}\n\nColumns: {names}")


def data_quality(profile: DatasetProfile) -> None:
    section_label("Data quality check")
    warnings = [i.message for i in profile.issues if i.severity == "warning"]
    notes = [i.message for i in profile.issues if i.severity == "info"]
    if not profile.issues:
        callout("No data-quality problems found", ["No duplicates, empty columns or suspicious values detected."], "info")
        return
    callout(f"Worth fixing ({len(warnings)})", warnings, "warn")
    if notes:
        with st.expander(f"Notes for context ({len(notes)})", expanded=not warnings):
            for n in notes:
                st.markdown(f"- {n}")


def relationships(profile: DatasetProfile) -> None:
    section_label("Columns that move together")
    if not profile.correlations:
        st.caption("No strong relationships found between numeric columns.")
        return
    for c in profile.correlations:
        direction = "rise together" if c.r > 0 else "move in opposite directions"
        st.markdown(f"- **{c.column_a}** and **{c.column_b}** {direction} "
                    f"({c.strength}, r = {c.r:+.2f}, {c.n:,} rows)")
    st.caption("r runs from −1 to +1. Correlation shows a pattern, not that one column causes the other.")


def overview_tab(profile: DatasetProfile, fmt: Formatter) -> None:
    dataset_metrics(profile)
    key_figures(profile, fmt)
    column_roles(profile)
    left, right = st.columns([3, 2], gap="large")
    with left:
        data_quality(profile)
    with right:
        relationships(profile)


# ---------------------------------------------------------------- columns tab

def headline_stat(c: ColumnProfile, fmt: Formatter) -> str:
    s = c.stats
    if not s:
        return "no values"
    if c.role == "measure":
        return f"avg {fmt.number(s['mean'], c.name)} · range {fmt.number(s['min'], c.name)} – {fmt.number(s['max'], c.name)}"
    if c.role in ("category", "boolean"):
        return f"most common: {s['top_value']} ({s['top_pct']}%)"
    if c.role == "date":
        return f"{s['min']} to {s['max']}"
    if c.role == "text":
        return f"avg length {s['avg_length']} characters"
    return "e.g. " + ", ".join(s.get("examples", [])[:2])


def column_detail(c: ColumnProfile, fmt: Formatter) -> None:
    name, desc = ROLE_INFO[c.role]
    st.markdown(f"#### {c.name}")
    st.caption(f"{name} — {desc}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Values", f"{c.count:,}")
    m2.metric("Missing", f"{c.missing:,}", help=f"{c.missing_pct}% of rows")
    m3.metric("Distinct values", f"{c.unique:,}")

    s = c.stats
    if not s:
        st.caption("This column has no values.")
    elif c.role == "measure":
        n = lambda k: fmt.number(s[k], c.name)  # noqa: E731
        table = pd.DataFrame({
            "Statistic": ["Total", "Average (mean)", "Median (middle value)", "Standard deviation", "Minimum",
                          "25th percentile", "75th percentile", "Maximum", "Negative values", "Zero values",
                          "Unusual values (outliers)"],
            "Value": [n("sum"), n("mean"), n("median"), n("std"), n("min"), n("q1"), n("q3"), n("max"),
                      fmt.count(s["negatives"]), fmt.count(s["zeros"]), fmt.count(s["outliers"])],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("The median is often more representative than the average when a few very large "
                   "values pull the average up.")
    elif c.role in ("category", "boolean"):
        top = pd.DataFrame(s["top_values"]).rename(columns={"value": "Value", "count": "Rows", "pct": "Share %"})
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "Rows": st.column_config.NumberColumn(format="localized"),
            "Share %": st.column_config.ProgressColumn("Share %", min_value=0, max_value=100, format="%.1f%%")})
        if c.unique > len(top):
            st.caption(f"Showing the top {len(top)} of {c.unique:,} values.")
    elif c.role == "date":
        d1, d2, d3 = st.columns(3)
        d1.metric("Earliest", s["min"])
        d2.metric("Latest", s["max"])
        d3.metric("Distinct days", f"{s['distinct_dates']:,}", help=f"Spanning {s['span_days']:,} days")
    else:
        st.write("Examples:", ", ".join(f"`{e}`" for e in s.get("examples", [])))


def columns_tab(profile: DatasetProfile, fmt: Formatter) -> None:
    summary = pd.DataFrame([{
        "Column": c.name, "Role": ROLE_INFO[c.role][0], "Missing %": c.missing_pct,
        "Distinct values": c.unique, "Summary": headline_stat(c, fmt),
    } for c in profile.columns])
    st.dataframe(summary, hide_index=True, width="stretch", height=min(35 * (len(summary) + 1) + 3, 600),
                 column_config={"Missing %": st.column_config.ProgressColumn(
                     "Missing %", min_value=0, max_value=100, format="%.1f%%"),
                     "Distinct values": st.column_config.NumberColumn(format="localized")})
    with st.popover("What do the roles mean?"):
        for name, desc in ROLE_INFO.values():
            st.markdown(f"**{name}** — {desc}")
    st.divider()
    chosen = st.selectbox("Inspect a column", [c.name for c in profile.columns])
    column_detail(profile.column(chosen), fmt)
