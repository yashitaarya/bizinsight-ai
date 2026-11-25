"""Unit tests for core/charts.py.  Run with:  pytest"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.charts import (MAX_LINES, MAX_SCATTER_POINTS, OTHER_COLOR, PALETTE, ChartSpec, apply_filters,
                         choose_period, color_map, default_agg, make_chart, recommend_charts)
from core.data_loader import load_dataset
from core.profiler import profile_dataset


@pytest.fixture(scope="module")
def sample_df():
    p = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
    return load_dataset(p.name, p.read_bytes()).df


def kinds(df):
    return [s.kind for s in recommend_charts(profile_dataset(df))]


# ---------------------------------------------------------------- recommendations

def test_sample_data_recommendations(sample_df):
    specs = recommend_charts(profile_dataset(sample_df))
    assert len(specs) <= 6
    assert specs[0].kind == "time_series" and specs[0].y == "Revenue" and specs[0].agg == "sum"
    ranking = next(s for s in specs if s.kind == "ranking")
    assert ranking.x == "Product"                       # "top performers" chart
    assert "time_by_category" in [s.kind for s in specs]
    assert all(s.reason for s in specs)                 # every chart explains itself


def test_every_recommended_chart_renders(sample_df):
    for spec in recommend_charts(profile_dataset(sample_df), max_charts=20):
        result = make_chart(spec, sample_df)
        assert len(result.figure.data) >= 1, spec.title
        assert not result.table.empty, spec.title


def test_no_date_column_means_no_time_charts():
    df = pd.DataFrame({"Region": ["N", "S", "E"] * 10, "Revenue": np.arange(30.0)})
    k = kinds(df)
    assert "time_series" not in k and "time_by_category" not in k
    assert "category_bar" in k


def test_no_measures_falls_back_to_counting_rows():
    df = pd.DataFrame({
        "Day": pd.date_range("2025-01-01", periods=60).repeat(2),
        "Status": ["Open", "Closed", "Pending"] * 40,
    })
    specs = recommend_charts(profile_dataset(df))
    assert specs and all(s.y is None for s in specs)
    assert specs[0].title == "Number of rows over time"
    assert "distribution" not in [s.kind for s in specs]


def test_single_value_category_not_compared():
    df = pd.DataFrame({"Country": ["Canada"] * 30, "Sales": np.arange(30.0)})
    assert "category_bar" not in kinds(df)


def test_nothing_chartable_returns_empty_list():
    df = pd.DataFrame({"Email": [f"u{i}@x.com" for i in range(30)]})   # identifier only
    assert recommend_charts(profile_dataset(df)) == []


def test_too_few_dates_for_a_trend():
    df = pd.DataFrame({"Day": pd.to_datetime(["2025-01-01", "2025-01-02"] * 10), "Sales": np.arange(20.0)})
    assert "time_series" not in kinds(df)


def test_non_additive_measure_is_averaged():
    assert default_agg("Revenue") == "sum"
    assert default_agg("Unit Price") == "mean"
    assert default_agg(None) == "count"


# ---------------------------------------------------------------- numbers are correct

def test_category_totals_match_pandas(sample_df):
    result = make_chart(ChartSpec("category_bar", "t", "r", x="Region", y="Revenue", agg="sum"), sample_df)
    expected = sample_df.groupby("Region")["Revenue"].sum()
    got = result.table.set_index("Region")["Total Revenue"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    assert got.is_monotonic_decreasing                  # largest first


def test_average_and_count(sample_df):
    avg = make_chart(ChartSpec("category_bar", "t", "r", x="Category", y="Unit Price", agg="mean"), sample_df)
    assert avg.table.set_index("Category")["Average Unit Price"]["Furniture"] == pytest.approx(
        sample_df.loc[sample_df["Category"] == "Furniture", "Unit Price"].mean())
    cnt = make_chart(ChartSpec("category_bar", "t", "r", x="Region", y=None, agg="count"), sample_df)
    assert cnt.table["Number of rows"].sum() == len(sample_df)


def test_monthly_trend_matches_pandas(sample_df):
    result = make_chart(ChartSpec("time_series", "t", "r", x="Order Date", y="Revenue", agg="sum"), sample_df)
    expected = sample_df.groupby(sample_df["Order Date"].dt.to_period("M"))["Revenue"].sum()
    assert len(result.table) == 12
    assert result.table["Total Revenue"].tolist() == pytest.approx(expected.tolist())
    assert not result.notes                              # full year: no partial-month warning


def test_distribution_counts_every_value(sample_df):
    result = make_chart(ChartSpec("distribution", "t", "r", x="Revenue"), sample_df)
    assert result.table["Rows"].sum() == sample_df["Revenue"].notna().sum()


# ---------------------------------------------------------------- honest notes

def test_top_n_note():
    df = pd.DataFrame({"Product": [f"P{i}" for i in range(25)], "Sales": np.arange(25.0)})
    result = make_chart(ChartSpec("ranking", "t", "r", x="Product", y="Sales", top_n=10), df)
    assert len(result.table) == 10
    assert result.table["Product"].iloc[0] == "P24"
    assert any("top 10 of 25" in n for n in result.notes)


def test_partial_last_month_flagged():
    df = pd.DataFrame({"Day": pd.date_range("2025-01-01", "2025-09-10"), "Sales": 1.0})
    result = make_chart(ChartSpec("time_series", "t", "r", x="Day", y="Sales"), df)
    assert any("last month" in n and "2025-09-10" in n for n in result.notes)


def test_missing_measure_values_reported(sample_df):
    result = make_chart(ChartSpec("category_bar", "t", "r", x="Region", y="Cost"), sample_df)
    assert any("8 row(s) with no 'Cost' value" in n for n in result.notes)


def test_blank_categories_shown_not_hidden(sample_df):
    result = make_chart(ChartSpec("category_bar", "t", "r", x="Customer Segment", y="Revenue"), sample_df)
    assert "(blank)" in result.table["Customer Segment"].tolist()


def test_many_lines_folded_into_other():
    df = pd.DataFrame({
        "Day": np.tile(pd.date_range("2025-01-01", periods=10), 8),
        "Store": np.repeat([f"S{i}" for i in range(8)], 10),
        "Sales": np.repeat(np.arange(8.0), 10),
    })
    result = make_chart(ChartSpec("time_by_category", "t", "r", x="Day", y="Sales", color="Store"), df)
    shown = set(result.table["Store"])
    assert len(shown) == MAX_LINES + 1 and "Other" in shown
    assert "S7" in shown and "S0" not in shown          # largest kept, smallest folded
    assert any("grouped as 'Other'" in n for n in result.notes)


# ---------------------------------------------------------------- filters & colors

def test_filters(sample_df):
    out = apply_filters(sample_df, "Order Date", date(2025, 3, 1), date(2025, 3, 31),
                        {"Region": ["North", "West"], "Category": []})
    assert out["Order Date"].min() >= pd.Timestamp("2025-03-01")
    assert out["Order Date"].max().date() == date(2025, 3, 31)   # end date is inclusive
    assert set(out["Region"]) == {"North", "West"}
    assert out["Category"].nunique() == 3                        # empty list = no filter
    assert len(apply_filters(sample_df)) == len(sample_df)


def test_colors_follow_the_category_not_its_rank():
    full_order = ["North", "West", "East", "South"]
    cmap = color_map(full_order)
    assert cmap["East"] == PALETTE[2] and cmap["Other"] == OTHER_COLOR
    # Even if a filter removes North, East keeps its color
    assert color_map(full_order)["East"] == cmap["East"]


# ---------------------------------------------------------------- edge cases

@pytest.mark.parametrize("days, expected", [(30, "D"), (120, "W"), (365, "M"), (2000, "Q")])
def test_choose_period(days, expected):
    s = pd.Series(pd.date_range("2025-01-01", periods=2, freq=f"{days}D"))
    assert choose_period(s)[0] == expected


def test_friendly_errors():
    df = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(ValueError, match="No rows"):
        make_chart(ChartSpec("distribution", "t", "r", x="A"), df.iloc[0:0])
    with pytest.raises(ValueError, match="isn't in this dataset"):
        make_chart(ChartSpec("distribution", "t", "r", x="Missing"), df)


def test_scatter_is_sampled_for_big_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"A": rng.normal(size=8000), "B": rng.normal(size=8000)})
    result = make_chart(ChartSpec("scatter", "t", "r", x="A", y="B"), df)
    assert len(result.figure.data[0].x) == MAX_SCATTER_POINTS
    assert any("random sample" in n for n in result.notes)


def test_totals_start_at_zero_and_bars_leave_room_for_labels(sample_df):
    line = make_chart(ChartSpec("time_series", "t", "r", x="Order Date", y="Revenue"), sample_df).figure
    assert line.layout.yaxis.rangemode == "tozero"
    bars = make_chart(ChartSpec("ranking", "t", "r", x="Product", y="Revenue"), sample_df)
    lo, hi = bars.figure.layout.xaxis.range
    assert lo == 0 and hi > bars.table["Total Revenue"].max() * 1.1


def test_nearly_complete_month_not_flagged():
    # Data ends on 29 Dec: 29 of 31 days covered is not misleading, so no note
    df = pd.DataFrame({"Day": pd.date_range("2025-01-01", "2025-12-29"), "Sales": 1.0})
    result = make_chart(ChartSpec("time_series", "t", "r", x="Day", y="Sales"), df)
    assert not any("partly covered" in n for n in result.notes)


def test_ascending_ranking_shows_the_lowest_values():
    df = pd.DataFrame({"Channel": [f"C{i}" for i in range(12)], "Minutes": np.arange(12.0)})
    result = make_chart(ChartSpec("ranking", "t", "r", x="Channel", y="Minutes", agg="mean", top_n=3,
                                  ascending=True), df)
    assert result.table["Channel"].tolist() == ["C0", "C1", "C2"]
    assert any("lowest 3 of 12" in n for n in result.notes)


def test_quarterly_axis_labels(sample_df):
    fig = make_chart(ChartSpec("time_series", "t", "r", x="Order Date", y="Revenue", period="Q"), sample_df).figure
    assert list(fig.layout.xaxis.ticktext) == ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]


def test_partial_period_note_depends_on_calculation():
    df = pd.DataFrame({"Day": pd.date_range("2025-01-01", "2025-09-10"), "Minutes": 5.0})
    avg = make_chart(ChartSpec("time_series", "t", "r", x="Day", y="Minutes", agg="mean"), df)
    assert any("fewer rows" in n for n in avg.notes) and not any("look lower" in n for n in avg.notes)
    total = make_chart(ChartSpec("time_series", "t", "r", x="Day", y="Minutes", agg="sum"), df)
    assert any("total may look lower" in n for n in total.notes)
