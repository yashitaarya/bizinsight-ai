"""Unit tests for core/profiler.py.  Run with:  pytest"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import load_dataset
from core.profiler import classify_column, profile_dataset, suggest_kpi_columns


# ---------------------------------------------------------------- column roles

@pytest.mark.parametrize("name, values, expected", [
    ("Revenue", [10.5, 20.0, 30.25] * 10, "measure"),
    ("Order ID", list(range(30)), "identifier"),                 # name says ID
    ("Ticket", list(range(1000, 1030)), "identifier"),           # all-unique whole numbers
    ("Units", list(range(1, 31)), "measure"),                    # unique but clearly additive
    ("Region", ["North", "South", "East"] * 10, "category"),
    ("Email", [f"user{i}@x.com" for i in range(30)], "identifier"),
    ("Active", [True, False] * 15, "boolean"),
    ("Date", pd.date_range("2025-01-01", periods=30), "date"),
])
def test_classify_column(name, values, expected):
    assert classify_column(name, pd.Series(values)) == expected


def test_free_text_column():
    comments = pd.Series([f"comment number {i % 100}" for i in range(300)])
    assert classify_column("Notes", comments) == "text"


# ---------------------------------------------------------------- statistics

@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "Region": ["North", "North", "South", "East", None],
        "Revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Day": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-10", "2025-01-31"]),
    })


def test_dataset_level_numbers(sales_df):
    p = profile_dataset(sales_df)
    assert (p.n_rows, p.n_columns) == (5, 3)
    assert p.missing_cells == 1
    assert p.completeness_pct == pytest.approx(100 - 1 / 15 * 100, abs=0.01)
    assert p.duplicate_rows == 0


def test_measure_stats_match_pandas(sales_df):
    s = profile_dataset(sales_df).column("Revenue").stats
    assert s["sum"] == 1500
    assert s["mean"] == 300
    assert s["median"] == 300
    assert s["min"] == 100 and s["max"] == 500
    assert s["std"] == pytest.approx(sales_df["Revenue"].std(), rel=1e-4)


def test_category_stats(sales_df):
    c = profile_dataset(sales_df).column("Region")
    assert c.missing == 1
    assert c.stats["top_value"] == "North"
    assert c.stats["top_count"] == 2
    assert c.stats["top_pct"] == 50.0          # 2 of the 4 non-missing values


def test_date_stats(sales_df):
    s = profile_dataset(sales_df).column("Day").stats
    assert (s["min"], s["max"], s["span_days"]) == ("2025-01-01", "2025-01-31", 30)


# ---------------------------------------------------------------- quality checks

def messages(profile):
    return " | ".join(i.message for i in profile.issues)


def test_duplicates_reported_as_warning():
    df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
    p = profile_dataset(df)
    assert p.duplicate_rows == 1
    assert p.issues[0].severity == "warning" and "duplicate" in p.issues[0].message


def test_mostly_empty_and_constant_columns():
    df = pd.DataFrame({"Sparse": [1, None, None, None], "Same": ["a"] * 4, "X": [1, 2, 3, 4]})
    text = messages(profile_dataset(df))
    assert "'Sparse' is mostly empty" in text
    assert "'Same' contains only one distinct value ('a')" in text


def test_outliers_detected():
    values = [10.0] * 20 + [11.0] * 20 + [12.0] * 20 + [1000.0]
    p = profile_dataset(pd.DataFrame({"Sales": values}))
    st = p.column("Sales").stats
    assert st["outliers"] == 1 and st["outliers_high"] == 1 and st["outliers_low"] == 0
    assert "1 unusually high" in messages(p)


def test_mostly_numeric_text_column_flagged():
    # The loader leaves this as text (only 80% numeric) — the profiler should explain why.
    df = pd.DataFrame({"Amount": ["10", "20", "30", "40", "50", "60", "70", "80", "N/A", "unknown"]})
    text = messages(profile_dataset(df))
    assert "'Amount' is mostly numbers" in text and "'N/A'" in text


def test_clean_data_has_no_warnings():
    df = pd.DataFrame({"Region": ["N", "S"] * 10, "Sales": np.arange(20, dtype=float)})
    assert not [i for i in profile_dataset(df).issues if i.severity == "warning"]


# ---------------------------------------------------------------- correlations & KPIs

def test_correlation_found_and_weak_ignored():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"Ads": x, "Sales": x * 3 + rng.normal(scale=0.1, size=200), "Noise": rng.normal(size=200)})
    corr = profile_dataset(df).correlations
    assert len(corr) == 1
    assert {corr[0].column_a, corr[0].column_b} == {"Ads", "Sales"}
    assert corr[0].strength == "strong" and corr[0].r > 0.9


def test_kpi_suggestions_prefer_additive_columns():
    df = pd.DataFrame({"Unit Price": [1.0, 2.0, 3.0], "Revenue": [5.0, 6.0, 7.0], "Discount": [0.1, 0.2, 0.0]})
    assert suggest_kpi_columns(profile_dataset(df)) == ["Revenue"]


def test_kpi_fallback_when_no_additive_names():
    df = pd.DataFrame({"Score": [1.0, 2.0, 3.0], "Rating": [4.0, 5.0, 3.0]})
    assert suggest_kpi_columns(profile_dataset(df)) == ["Score", "Rating"]


# ---------------------------------------------------------------- robustness

def test_profile_is_json_serialisable():
    sample = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
    df = load_dataset(sample.name, sample.read_bytes()).df
    json.dumps(profile_dataset(df).to_dict())   # raises if any numpy/Timestamp slipped through


def test_handles_all_missing_and_single_row():
    df = pd.DataFrame({"Empty": [None], "Num": [5.0], "Txt": ["a"]})
    p = profile_dataset(df)
    assert p.n_rows == 1
    assert p.column("Num").stats["sum"] == 5


def test_mostly_unique_labels_are_text_not_category():
    # 49 different values in 62 rows is not a useful grouping column
    s = pd.Series([f"value {i}" for i in range(49)] + ["value 0"] * 13)
    assert classify_column("Budget", s) == "text"


def test_kpis_ordered_by_business_importance():
    df = pd.DataFrame({"Units": [1.0, 2.0], "Cost": [3.0, 4.0], "Revenue": [5.0, 6.0]})
    assert suggest_kpi_columns(profile_dataset(df)) == ["Revenue", "Cost", "Units"]
