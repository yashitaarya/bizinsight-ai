"""Unit tests for core/facts.py — the numbers the AI is allowed to use."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import load_dataset
from core.facts import build_fact_sheet, change, num
from core.insights import build_request
from core.profiler import profile_dataset


@pytest.fixture(scope="module")
def sample():
    p = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
    ds = load_dataset(p.name, p.read_bytes())
    profile = profile_dataset(ds.df)
    return ds, profile, build_fact_sheet(ds.df, profile, ds.cleaning_log)


def text_of(facts):
    return "\n".join(f.text for f in facts)


def test_ids_are_unique_and_sequential(sample):
    _, _, facts = sample
    assert [f.id for f in facts] == [f"F{i}" for i in range(1, len(facts) + 1)]
    assert {f.category for f in facts} <= {"overview", "kpi", "trend", "segment", "relationship", "quality"}


def test_numbers_match_pandas(sample):
    ds, _, facts = sample
    df, text = ds.df, text_of(facts)
    assert num(df["Revenue"].sum()) in text
    laptop = df.loc[df["Product"] == "Laptop", "Revenue"].sum()
    assert f"Laptop ({num(laptop)}, {laptop / df['Revenue'].sum() * 100:.1f}% of the total)" in text
    both = df[["Revenue", "Cost"]].dropna()
    assert f"Revenue minus Cost is {num(both['Revenue'].sum() - both['Cost'].sum())}" in text


def test_trend_uses_complete_months_only():
    # Data from 15 Jan to 10 Sep (grouped by month): January and September are partial and must be left out.
    days = pd.date_range("2025-01-15", "2025-09-10")
    df = pd.DataFrame({"Day": days, "Sales": 100.0})
    profile = profile_dataset(df)
    trend = [f.text for f in build_fact_sheet(df, profile) if f.category == "trend"]
    assert trend and "complete months (2025-02 to 2025-08)" in trend[0]


def test_trend_change_is_correct():
    df = pd.DataFrame({"Day": pd.date_range("2025-01-01", "2025-08-31"), "Sales": 1.0})
    df.loc[df["Day"] >= "2025-08-01", "Sales"] = 2.0          # August doubles per day
    text = text_of(build_fact_sheet(df, profile_dataset(df)))
    jan, aug = 31.0, 62.0
    assert f"last complete month (2025-08) was 62, {(aug - jan) / jan * 100:.1f}% higher" in text


def test_no_dates_no_trend_and_no_measures_counts_rows():
    df = pd.DataFrame({"Status": ["Open", "Closed", "Open", "Pending"] * 10})
    facts = build_fact_sheet(df, profile_dataset(df))
    assert not [f for f in facts if f.category == "trend"]
    seg = next(f.text for f in facts if f.category == "segment")
    assert seg.startswith("Number of rows by Status") and "Open (20, 50.0% of the total)" in seg


def test_averages_not_totals_for_non_additive_measures():
    df = pd.DataFrame({"Team": ["A", "B"] * 10, "Satisfaction": np.tile([3.0, 4.0, 5.0, 4.0, 2.0], 4)})
    text = text_of(build_fact_sheet(df, profile_dataset(df)))
    assert "Average Satisfaction" in text and "Total Satisfaction" not in text
    assert "% of the total" not in text      # shares of an average make no sense


def test_quality_issues_and_cleaning_log_included(sample):
    _, _, facts = sample
    quality = text_of([f for f in facts if f.category == "quality"])
    assert "'Customer Segment' has 15 missing" in quality
    assert "Converted 'Unit Price' from text to numbers" in quality


def test_change_helper():
    assert change(120, 100) == "20.0% higher than"
    assert change(80, 100) == "20.0% lower than"
    assert change(5, 0) is None


# ---------------------------------------------------------------- privacy

def test_request_contains_no_raw_rows(sample):
    ds, profile, facts = sample
    request = build_request(facts, profile, "sales.csv")
    payload = json.loads(request)
    assert set(payload) == {"dataset", "columns", "facts"}
    assert "ORD-" not in request                            # no order IDs
    assert "$78.36" not in request and "78.36" not in request   # no raw cell values
    assert len(request) < 12_000                            # a summary, not a data dump
    assert all(set(c) == {"name", "type"} for c in payload["columns"])
