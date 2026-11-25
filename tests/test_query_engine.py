"""Unit tests for core/query_engine.py — the safety boundary between the AI and the data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import load_dataset
from core.profiler import profile_dataset
from core.query_engine import QueryError, run_query, validate_plan


@pytest.fixture(scope="module")
def sample():
    p = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
    df = load_dataset(p.name, p.read_bytes()).df
    return df, profile_dataset(df)


def run(sample, **plan):
    df, profile = sample
    return run_query(validate_plan(plan, df, profile), df)


def texts(result):
    return " | ".join(f.text for f in result.facts)


# ---------------------------------------------------------------- correct numbers

def test_aggregate_total(sample):
    df, _ = sample
    r = run(sample, operation="aggregate", metric="Revenue", aggregation="sum")
    assert r.table["Value"].iloc[0] == pytest.approx(df["Revenue"].sum())
    assert r.facts[0].id == "Q1" and "1,200 rows" in r.facts[0].text


def test_aggregate_with_filters(sample):
    df, _ = sample
    r = run(sample, operation="aggregate", metric="Unit Price", aggregation="mean",
            filters=[{"column": "Category", "values": ["Furniture"]}],
            date_from="2025-06-01", date_to="2025-06-30")
    mask = (df["Category"] == "Furniture") & df["Order Date"].between("2025-06-01", "2025-06-30")
    assert r.table["Value"].iloc[0] == pytest.approx(df.loc[mask, "Unit Price"].mean())
    assert r.rows_used == mask.sum()
    assert "Category is Furniture" in r.filter_text and "2025-06-01 to 2025-06-30" in r.filter_text


def test_group_by_matches_pandas_with_shares(sample):
    df, _ = sample
    r = run(sample, operation="group_by", metric="Revenue", aggregation="sum", group_by="Region")
    expected = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    assert r.table["Region"].tolist() == expected.index.tolist()
    assert r.table["Total Revenue"].tolist() == pytest.approx(expected.tolist())
    assert r.table["Share of total %"].sum() == pytest.approx(100, abs=0.2)
    assert "#1 highest: Region 'North'" in r.facts[0].text
    assert r.chart is not None and r.chart.x == "Region"


def test_group_by_lowest_and_limit(sample):
    r = run(sample, operation="group_by", metric="Revenue", group_by="Product", sort="asc", limit=2)
    assert len(r.table) == 2 and r.table["Product"].iloc[0] == "Pen Set"
    assert "lowest" in r.facts[0].text and "Only 2 are listed" in texts(r)


def test_profit_via_subtract_metric(sample):
    df, _ = sample
    r = run(sample, operation="group_by", metric="Revenue", subtract_metric="Cost", group_by="Product", limit=1)
    laptop = df[df["Product"] == "Laptop"]
    assert r.table.iloc[0, 1] == pytest.approx((laptop["Revenue"] - laptop["Cost"]).sum())
    assert "Revenue minus Cost" in r.title


def test_time_trend_by_quarter(sample):
    df, _ = sample
    r = run(sample, operation="time_trend", metric="Revenue", aggregation="sum", period="quarter")
    expected = df.groupby(df["Order Date"].dt.to_period("Q"))["Revenue"].sum()
    assert r.table.iloc[:, 1].tolist() == pytest.approx(expected.tolist())
    assert r.chart.period == "Q"
    assert "first complete quarter" in texts(r)


def test_partial_periods_flagged():
    df = pd.DataFrame({"Day": pd.date_range("2025-01-20", "2025-05-05"), "Sales": 1.0})
    profile = profile_dataset(df)
    r = run_query(validate_plan({"operation": "time_trend", "metric": "Sales", "period": "month"}, df, profile), df)
    assert r.table["Complete?"].tolist() == ["no", "yes", "yes", "yes", "no"]
    assert "2025-01" in texts(r) and "only partly covered" in r.facts[0].text


def test_count_rows_and_numeric_filter(sample):
    df, _ = sample
    r = run(sample, operation="aggregate", aggregation="count",
            numeric_filters=[{"column": "Revenue", "op": ">", "value": 5000}])
    assert r.table["Value"].iloc[0] == (df["Revenue"] > 5000).sum()


def test_correlation_and_count_distinct(sample):
    r = run(sample, operation="correlation", metric="Revenue", second_metric="Cost")
    assert "strong positive correlation" in r.facts[0].text
    r = run(sample, operation="count_distinct", group_by="Product")
    assert "'Product' has 9 different values" in r.facts[0].text


def test_filter_values_are_case_insensitive_but_exact(sample):
    r = run(sample, operation="aggregate", metric="Revenue",
            filters=[{"column": "region", "values": ["north"]}])
    assert r.filter_text == "Region is North"


# ---------------------------------------------------------------- rejected plans

@pytest.mark.parametrize("plan, message", [
    ({"operation": "exec", "metric": "Revenue"}, "isn't one of the supported"),
    ({"operation": "aggregate", "metric": "Revenue", "aggregation": "__import__('os')"}, "isn't a supported calculation"),
    ({"operation": "aggregate", "metric": "Profit"}, "no column called 'Profit'"),
    ({"operation": "aggregate", "metric": "Region"}, "it's a category column"),
    ({"operation": "group_by", "metric": "Revenue", "group_by": "Order ID"}, "it's an identifier column"),
    ({"operation": "group_by", "metric": "Revenue"}, "needs a category column"),
    ({"operation": "aggregate", "metric": "Revenue", "filters": [{"column": "Region", "values": ["Mars"]}]},
     "no value called 'Mars'"),
    ({"operation": "aggregate", "metric": "Revenue", "date_from": "next tuesday"}, "isn't a valid start date"),
    ({"operation": "aggregate", "metric": "Revenue", "date_from": "2025-06-01", "date_to": "2025-01-01"},
     "start date is after"),
    ({"operation": "aggregate", "metric": "Revenue",
      "numeric_filters": [{"column": "Revenue", "op": "; DROP TABLE", "value": 1}]}, "supported comparison"),
    ({"operation": "correlation", "metric": "Revenue", "second_metric": "Revenue"}, "two different"),
    ({"operation": "time_trend", "metric": "Revenue", "period": "fortnight"}, "time period"),
])
def test_invalid_plans_are_rejected(sample, plan, message):
    df, profile = sample
    with pytest.raises(QueryError, match=message):
        validate_plan(plan, df, profile)


def test_time_question_without_date_column():
    df = pd.DataFrame({"Team": ["A", "B"] * 10, "Score": np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 4)})
    with pytest.raises(QueryError, match="no date column"):
        validate_plan({"operation": "time_trend", "metric": "Score"}, df, profile_dataset(df))


def test_filters_that_match_nothing(sample):
    with pytest.raises(QueryError, match="No rows match"):
        run(sample, operation="aggregate", metric="Revenue", date_from="2030-01-01")


def test_limit_is_clamped(sample):
    df, profile = sample
    assert validate_plan({"operation": "group_by", "group_by": "Region", "limit": 10_000}, df, profile).limit == 50
    assert validate_plan({"operation": "group_by", "group_by": "Region", "limit": -3}, df, profile).limit == 1


def test_no_metric_means_counting(sample):
    r = run(sample, operation="group_by", group_by="Region")
    assert "Number of rows" in r.title and r.table["Number of rows"].sum() == 1200
