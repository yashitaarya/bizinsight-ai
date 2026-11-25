"""Tests for the presentation layer: number formatting, currency detection and KPI highlights."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import load_dataset
from core.formatting import Formatter, compact, looks_like_money, period_label
from core.highlights import summary_highlights
from core.profiler import profile_dataset
from core.query_engine import run_query, validate_plan

SAMPLES = Path(__file__).parent.parent / "sample_data"


def load(name):
    p = SAMPLES / name
    ds = load_dataset(p.name, p.read_bytes())
    return ds, profile_dataset(ds.df)


# ---------------------------------------------------------------- formatting

@pytest.mark.parametrize("value, text", [
    (2_602_869.24, "2.60M"), (3_400_000_000, "3.40B"), (842_551.57, "842,552"), (12_124, "12,124"),
    (1_500, "1,500"), (893.93, "893.93"), (12, "12"), (-1_234.5, "-1,234.50"), (0, "0"),
])
def test_compact(value, text):
    assert compact(value) == text


def test_money_only_with_a_detected_symbol():
    assert Formatter("$").number(2_602_869.24, "Revenue") == "$2.60M"
    assert Formatter("€").number(-1_234.5, "Cost") == "-€1,234.50"
    assert Formatter("$").number(12_124, "Units") == "12,124"          # not money
    assert Formatter(None).number(2_602_869.24, "Revenue") == "2.60M"  # no symbol in data -> none invented
    assert Formatter("$").number(None) == "—"


def test_money_column_detection():
    assert looks_like_money("Total Revenue") and looks_like_money("Unit Price") and looks_like_money("Salary")
    assert not looks_like_money("Units") and not looks_like_money("Discount") and not looks_like_money("Rate")


@pytest.mark.parametrize("code, text", [("D", "21 May 2025"), ("W", "Week of 19 May 2025"), ("M", "May 2025"),
                                        ("Q", "Q2 2025"), ("Y", "2025")])
def test_period_labels(code, text):
    assert period_label(pd.Timestamp("2025-05-21").to_period(code)) == text


def test_currency_detected_from_data():
    ds, _ = load("sales_sample.csv")
    assert ds.currency_symbol == "$"
    ds2, _ = load("support_tickets.csv")
    assert ds2.currency_symbol is None


# ---------------------------------------------------------------- summary highlights

def test_sales_highlights():
    ds, profile = load("sales_sample.csv")
    hl = summary_highlights(ds.df, profile, Formatter(ds.currency_symbol))
    labels = [h.label for h in hl]
    assert labels == ["Total Revenue", "Total Revenue, Dec 2025", "Revenue minus Cost", "Highest Product"]
    assert hl[0].value == "$2.60M"
    assert hl[1].note == "+23.8% vs Jan 2025" and hl[1].trend == "up"
    assert hl[2].note == "32.7% of Revenue"
    assert hl[3].value == "Laptop" and "45.0% of total" in hl[3].note


def test_highlights_adapt_to_non_sales_data():
    ds, profile = load("support_tickets.csv")
    hl = summary_highlights(ds.df, profile, Formatter(ds.currency_symbol))
    assert hl[0].label.startswith("Average ")                        # times are averaged, not totaled
    assert not any("$" in h.value or "$" in h.note for h in hl)
    assert all("Total" not in h.label for h in hl)


def test_highlights_with_only_categories():
    df = pd.DataFrame({"Status": ["Open", "Closed", "Open", "Pending"] * 10})
    hl = summary_highlights(df, profile_dataset(df), Formatter())
    assert hl[0].label == "Rows" and hl[0].value == "40"
    assert any(h.label == "Highest Status" and h.value == "Open" for h in hl)


def test_highlights_with_only_numbers():
    df = pd.DataFrame({"Score": np.tile([1.0, 2.0, 3.0, 4.0], 10)})
    hl = summary_highlights(df, profile_dataset(df), Formatter())
    assert hl[0].label == "Average Score" and len(hl) >= 1


# ---------------------------------------------------------------- query result highlights

def run(ds, profile, **plan):
    return run_query(validate_plan(plan, ds.df, profile), ds.df, Formatter(ds.currency_symbol))


def test_group_by_highlights_are_formatted():
    ds, profile = load("sales_sample.csv")
    r = run(ds, profile, operation="group_by", metric="Revenue", subtract_metric="Cost", group_by="Product")
    first, second, groups = r.highlights
    assert (first.label, first.value) == ("Highest Product", "Laptop")
    assert first.note == "$384,001 · 45.6% of total"
    assert second.label == "Runner-up Product" and groups.value == "9"


def test_trend_highlights_use_complete_periods():
    ds, profile = load("support_tickets.csv")     # data ends mid-October -> October is partial
    r = run(ds, profile, operation="time_trend", metric="Resolution Hours", aggregation="mean", period="month")
    latest = r.highlights[0]
    assert latest.label == "Latest complete month" and latest.note == "Sep 2025"
    assert r.table["Complete?"].iloc[-1] == "no"
    assert any(h.label == "Change" and h.value.endswith("%") for h in r.highlights)


def test_correlation_and_count_highlights():
    ds, profile = load("support_tickets.csv")
    r = run(ds, profile, operation="correlation", metric="Resolution Hours", second_metric="Satisfaction Score")
    assert r.highlights[0].label == "Correlation (r)" and r.highlights[0].value.startswith("-")
    r = run(ds, profile, operation="aggregate", aggregation="count")
    assert r.highlights[0].value == "1,500"


# ---------------------------------------------------------------- homepage dashboard metrics

from core.highlights import dashboard_metrics, record_noun, status_column  # noqa: E402


def labels(df, currency=None):
    return [(h.label, h.value) for h in dashboard_metrics(df, profile_dataset(df), Formatter(currency))]


def test_dashboard_sales_metrics():
    ds, profile = load("sales_sample.csv")
    cards = dashboard_metrics(ds.df, profile, Formatter(ds.currency_symbol))
    got = {h.label: h for h in cards}
    assert [h.label for h in cards][:4] == ["Orders", "Total Revenue", "Avg Revenue per order", "Revenue minus Cost"]
    assert got["Orders"].value == "1,200"
    assert got["Avg Revenue per order"].value == Formatter("$").number(ds.df["Revenue"].sum() / 1200, "Revenue")
    assert "estimate" in got["Revenue minus Cost"].note
    assert len(cards) <= 5


def test_dashboard_support_metrics():
    ds, profile = load("support_tickets.csv")
    cards = dashboard_metrics(ds.df, profile, Formatter(ds.currency_symbol))
    names = [h.label for h in cards]
    assert names[0] == "Tickets" and "Avg Resolution Hours" in names
    assert any(n.endswith(" mix") for n in names)
    assert not any(n.startswith("Total") or "minus" in n for n in names)
    sat = next(h for h in cards if h.label == "Avg Satisfaction Score")
    assert "missing" in sat.note                                       # gaps are surfaced, not hidden


def test_profit_estimate_only_with_revenue_and_cost():
    df = pd.DataFrame({"Order ID": [f"O{i}" for i in range(30)], "Revenue": np.arange(30.0) + 1})
    assert not any("minus" in label for label, _ in labels(df))


def test_record_noun_from_id_column():
    def noun(col):
        return record_noun(profile_dataset(pd.DataFrame({col: [f"X{i}" for i in range(30)]})))
    assert noun("Order ID") == "Orders"
    assert noun("ticket_id") == "Tickets"
    assert noun("Invoice Number") == "Invoices"
    assert noun("Address") == "Records"                               # not an ID name -> generic


def test_status_column_prefers_status_like_names():
    df = pd.DataFrame({"Region": ["N", "S", "E"] * 10, "Status": ["Open", "Closed"] * 15, "Amount": 1.0})
    assert status_column(profile_dataset(df)).name == "Status"
    df2 = pd.DataFrame({"Region": ["N", "S", "E"] * 10, "Amount": 1.0})
    assert status_column(profile_dataset(df2)).name == "Region"       # fallback: smallest category


def test_dashboard_without_numbers_or_ids():
    df = pd.DataFrame({"Day": pd.date_range("2025-01-01", periods=120), "Priority": ["Low", "High", "Mid"] * 40})
    got = labels(df)
    assert got[0] == ("Records", "120")
    assert any(label == "Records per week" for label, _ in got)
    assert any(label == "Priority mix" for label, _ in got)


def test_dashboard_minimal_dataset():
    got = labels(pd.DataFrame({"Notes": [f"text {i}" for i in range(5)]}))
    assert got == [("Records", "5")]                                  # nothing invented


# ---------------------------------------------------------------- tidying AI-written text

from core.facts import Fact  # noqa: E402
from core.formatting import tidy_ai_text  # noqa: E402
from core.insights import unverified_numbers  # noqa: E402


@pytest.mark.parametrize("raw, tidy", [
    ("Total Revenue was 2,602,869.24 (F1, F2, F3).", "Total Revenue was 2.60M."),
    ("No proof of cause (see correlations F16, F17, F18).", "No proof of cause."),
    ("Last month (2025-12) was 243,851.67, up from (2025-01, 196,935.21).",
     "Last month (Dec 2025) was 243,852, up from (Jan 2025, 196,935)."),
    ("2025Q1 $187,503.75; peak in 2025Q3.", "Q1 2025 $187,504; peak in Q3 2025."),
    ("Average 2,169.06 vs median 893.93 across 1,200 rows in 2025.",
     "Average 2,169.06 vs median 893.93 across 1,200 rows in 2025."),        # short numbers untouched
    ("Covers 2025-01-01 to 2025-12-31.", "Covers 2025-01-01 to 2025-12-31."),  # full dates untouched
])
def test_tidy_ai_text(raw, tidy):
    assert tidy_ai_text(raw) == tidy


def test_tidy_keeps_quarters_that_are_not_fact_ids():
    assert tidy_ai_text("Profit 384,001.29 (Q1).", {"Q1"}) == "Profit 384,001."
    assert tidy_ai_text("Strongest in the final quarter (Q4).", {"Q1"}) == "Strongest in the final quarter (Q4)."


def test_requested_rounding_still_passes_the_number_check():
    facts = [Fact("F1", "kpi", "Total Revenue is 2,602,869.24; last month 243,851.67.")]
    assert unverified_numbers("Revenue was 2.6M, with about 244K last month.", facts) == []


def test_markdown_dollar_signs_are_escaped():
    from ui.components import md_safe
    assert md_safe("$1,200 and $3,400") == "\\$1,200 and \\$3,400"


def test_comparison_chart_shows_context_for_top_one_questions():
    ds, profile = load("sales_sample.csv")
    r = run(ds, profile, operation="group_by", metric="Revenue", group_by="Product", limit=1)
    assert len(r.table) == 1 and r.chart.top_n == 5
