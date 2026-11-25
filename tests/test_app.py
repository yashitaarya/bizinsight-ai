"""Tests that run the real Streamlit app headlessly (no browser needed).

AI features are tested against tests/fake_openai.py — a local fake server — so no API key
is needed and nothing is ever sent to OpenAI.
"""

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.fake_openai import FakeOpenAI, error_body, response_body

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture(autouse=True)
def never_call_real_openai(monkeypatch):
    """Safety net: tests must never use your real key from .env (that would cost money).
    An empty variable stops python-dotenv loading the real value, and the base URL points nowhere."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AI_DAILY_LIMIT", "1000")   # most tests check the per-session limit
    st.cache_resource.clear()   # fresh app-wide daily AI counter for every test


def run_with_sample(sample: str = "Retail sales") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.selectbox(key="sample").set_value(sample).run()
    assert not at.exception, at.exception
    return at


def page_text(at: AppTest) -> str:
    parts = ([m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning]
             + [e.label for e in at.expander])
    return "\n".join(parts)


def start_with_fake(monkeypatch, fake, sample: str = "Retail sales", page: str = "AI Summary") -> AppTest:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", fake.base_url)
    return go(run_with_sample(sample), page)


def button(at: AppTest, key: str):
    return next(b for b in at.button if b.key == key)


def go(at: AppTest, page: str) -> AppTest:
    """Switch page with the navigation bar."""
    at.session_state["nav"] = page
    at.run()
    assert not at.exception, at.exception
    return at


# ---------------------------------------------------------------- basics (Stages 1–3)

def test_welcome_screen():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert "BizInsight AI" in page_text(at)
    button(at, "welcome_tickets").click().run()                       # one-click sample
    assert not at.exception and at.session_state["sample"] == "Customer support tickets"
    assert at.session_state["nav"] == "Home"


def test_sample_data_overview():
    at = go(run_with_sample(), "Overview")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "1,200" and metrics["Completeness"] == "99.8%"
    assert metrics["Total Revenue"] == "$2.60M"                       # currency detected from the data
    assert not at.error


def test_key_figures_selection_and_column_inspector():
    at = go(run_with_sample(), "Overview")
    next(m for m in at.multiselect if m.label == "Columns to summarize").set_value(["Unit Price"]).run()
    assert "Average Unit Price" in {m.label for m in at.metric}      # prices are averaged, not totaled
    go(at, "Data")
    for col in ["Region", "Revenue", "Order Date", "Order ID"]:
        next(s for s in at.selectbox if s.label == "Inspect a column").set_value(col).run()
        assert not at.exception, f"{col}: {at.exception}"


def test_charts_tab_and_filters():
    at = go(run_with_sample(), "Charts")
    assert "Recommended charts" in [s.value for s in at.subheader]
    next(m for m in at.multiselect if m.label == "Region").set_value(["North"]).run()
    assert not at.exception
    assert "Showing **369** of 1,200 rows." in page_text(at)


def test_builder_chart_types():
    at = go(run_with_sample(), "Charts")
    options = next(r for r in at.radio if r.label == "Chart type").options
    assert options == ["Trend over time", "Compare categories", "Spread of values", "Two numbers together"]
    for option in options:
        next(r for r in at.radio if r.label == "Chart type").set_value(option).run()
        assert not at.exception, option


def test_second_sample_dataset_works_without_money_or_totals():
    at = go(run_with_sample("Customer support tickets"), "Overview")
    labels = {m.label for m in at.metric}
    assert "Average Resolution Hours" in labels and "Total Resolution Hours" not in labels
    assert "Amounts in" not in page_text(go(at, "Home"))              # no currency invented
    assert not at.error


# ---------------------------------------------------------------- AI Summary tab

def summary_json(**extra) -> str:
    item = lambda title, detail, *ids: {"title": title, "detail": detail, "fact_ids": list(ids)}  # noqa: E731
    data = {
        "headline": "Laptops drive almost half of revenue",
        "overview": "A full year of retail orders.",
        "key_findings": [item("Laptops lead revenue", "Laptops bring in 45% of all revenue.", "F12")],
        "trends": [item("Second half was stronger", "Revenue grew 9.3% in the second half.", "F11")],
        "business_takeaways": [item("Revenue is concentrated", "A few products carry most sales.", "F12")],
        "recommendations": [
            item("Review dependence on laptops", "Check exposure to one product line.", "F12"),
            item("Grow revenue next quarter", "Aim for 12.5% growth.", "F11"),     # made-up number
            item("Uncited advice", "No source.", ),                                 # should be removed
        ],
        "data_quality": [item("Data is nearly complete", "99.8% of cells are filled in.", "F19")],
        "limitations": ["The data cannot explain why sales changed."],
    }
    data.update(extra)
    return json.dumps(data)


def test_summary_tab_without_key_shows_setup_help_and_kpis():
    at = go(run_with_sample(), "AI Summary")
    text = page_text(at)
    assert "Connect OpenAI to turn on AI features" in text
    assert "Total Revenue" in text and "$2.60M" in text               # KPI cards work without AI
    assert not [b for b in at.button if b.key == "ai_generate"]


def test_ai_summary_generated_and_validated(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(summary_json()))
        at = start_with_fake(monkeypatch, fake)
        button(at, "ai_generate").click().run()
        assert not at.exception, at.exception

        sent = fake.requests[-1]
        assert sent["store"] is False and "ORD-" not in sent["input"]     # no raw IDs sent
        text = page_text(at)
        assert "Laptops drive almost half of revenue" in text              # headline
        assert "Laptops lead revenue" in text and "Business takeaways".upper() in text.upper()
        assert "Uncited advice" not in text                                # removed
        assert "1 statement(s) were removed" in text
        assert "Figure not verified" in text                               # 12.5% flagged
        assert "Sources & calculations" in text
        assert "F12: Total Revenue by Product" in text                     # references live in the expander
        assert "9 of 10 AI requests left" in text
        assert "Regenerate" in button(at, "ai_generate").label


def test_ai_summary_error_is_shown_not_crashed(monkeypatch):
    with FakeOpenAI() as fake:
        for _ in range(3):   # more errors than needed: the client doesn't retry
            fake.queue(429, error_body("You exceeded your current quota", "insufficient_quota", "insufficient_quota"))
        at = start_with_fake(monkeypatch, fake)
        button(at, "ai_generate").click().run()
        assert not at.exception
        assert "run out of credit" in page_text(at)


def test_session_request_limit(monkeypatch):
    with FakeOpenAI() as fake:
        at = start_with_fake(monkeypatch, fake)
        at.session_state["ai_requests_used"] = 10
        at.run()
        assert button(at, "ai_generate").disabled
        assert "AI request limit reached" in page_text(at)
        assert fake.requests == []


def test_malicious_text_is_escaped(monkeypatch):
    """Column names and AI text are shown as text, never interpreted as HTML."""
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(summary_json(headline="<img src=x onerror=alert(1)> Revenue up")))
        at = start_with_fake(monkeypatch, fake)
        button(at, "ai_generate").click().run()
        html = "\n".join(m.value for m in at.markdown)
        assert "<img src=x" not in html and "&lt;img src=x" in html


# ---------------------------------------------------------------- Ask tab

def plan_json(**fields) -> str:
    base = {"answerable": True, "interpretation": "Total revenue for each region", "reason_if_not": None,
            "suggested_questions": [], "operation": "group_by", "metric": "Revenue", "subtract_metric": None,
            "aggregation": "sum", "group_by": "Region", "date_column": None, "period": "auto",
            "second_metric": None, "sort": "desc", "limit": None, "filters": [], "date_from": None,
            "date_to": None, "numeric_filters": []}
    base.update(fields)
    return json.dumps(base)


def answer_json(text: str, ids: list[str], headline="North is the top region", takeaway="") -> str:
    return json.dumps({"headline": headline, "answer": text, "takeaway": takeaway, "fact_ids": ids, "caveats": []})


def test_ask_question_shows_structured_answer(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(plan_json()))
        fake.queue(200, response_body(answer_json("North sells the most: 799,640.77 (30.7% of revenue).", ["Q1"],
                                                  takeaway="Regional results are uneven.")))
        at = start_with_fake(monkeypatch, fake, page="Ask AI")
        at.chat_input[0].set_value("Which region sells the most?").run()
        assert not at.exception, at.exception

        assert len(fake.requests) == 2                                   # plan + answer
        assert "ORD-" not in json.dumps(fake.requests)                   # no IDs or rows sent
        text = page_text(at)
        assert "North is the top region" in text                         # headline
        assert "Highest Region" in text and "$799,641" in text           # KPI card, currency formatted
        assert "Business takeaway" in text and "Regional results are uneven." in text
        assert "Sources & calculations" in text and "Q1: #1 highest: Region 'North'" in text
        assert "(Q1)" not in text                                        # references are not inline any more
        assert any("North" in str(df.value) for df in at.dataframe)       # calculation table shown
        assert "9 of 10 AI requests left" in text


def test_example_question_button(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(plan_json(operation="aggregate", group_by=None)))
        fake.queue(200, response_body(answer_json("Total revenue is about 2.6 million.", ["Q1"],
                                                  headline="Total revenue is about 2.6 million")))
        at = start_with_fake(monkeypatch, fake, page="Ask AI")
        button(at, "example_What is the total Revenue?").click().run()
        assert not at.exception
        assert "Total revenue is about 2.6 million." in page_text(at)


def test_unanswerable_question_offers_alternatives(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(plan_json(
            answerable=False, operation="none", reason_if_not="The data shows what happened, not why.",
            suggested_questions=["How has Revenue changed month by month?"])))
        at = start_with_fake(monkeypatch, fake, page="Ask AI")
        at.chat_input[0].set_value("Why did sales drop in March?").run()
        assert not at.exception
        text = page_text(at)
        assert "can&#x27;t be answered from your dataset" in text or "can't be answered from your dataset" in text
        assert "not why" in text and "Try one of these instead" in text
        assert button(at, "alt_0_0").label == "How has Revenue changed month by month?"
        assert len(fake.requests) == 1


def test_ask_api_error_shown(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(401, error_body("Incorrect API key", "invalid_api_key"))
        at = start_with_fake(monkeypatch, fake, page="Ask AI")
        at.chat_input[0].set_value("Total revenue?").run()
        assert not at.exception
        assert "rejected the API key" in page_text(at)



# ---------------------------------------------------------------- Home page

def kpi_labels(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown if "bi-kpis" in m.value)


def test_home_sales_dashboard():
    at = run_with_sample()
    assert at.session_state["nav"] == "Home"
    text = page_text(at)
    assert "sales_sample.csv" in text and "Sample dataset loaded" in text
    assert "CSV" in text and "1,200 records" in text and "2025-01-01 to 2025-12-31" in text and "Amounts in $" in text
    kpis = kpi_labels(at)
    for label in ["Orders", "Total Revenue", "Avg Revenue per order", "Revenue minus Cost"]:
        assert label in kpis
    assert "$2.60M" in kpis and "(estimate)" in kpis
    assert len(at.get("plotly_chart")) == 2                            # trend + ranking
    assert "Data health" in text


def test_home_support_dashboard_uses_different_metrics():
    at = run_with_sample("Customer support tickets")
    kpis = kpi_labels(at)
    assert "Tickets" in kpis and "Avg Resolution Hours" in kpis and "mix" in kpis
    assert "Total" not in kpis and "$" not in kpis and "minus" not in kpis   # nothing sales-specific invented


def test_home_shows_no_ai_highlights_until_generated():
    at = run_with_sample()
    text = page_text(at)
    assert "AI features need an OpenAI API key" in text                 # no key: setup prompt, no insights
    assert "Key findings".upper() not in text.upper()


def test_home_generate_button_creates_summary_and_highlights(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(summary_json()))
        at = start_with_fake(monkeypatch, fake, page="Home")
        assert "Generate an AI summary to see highlights here" in page_text(at)
        button(at, "home_generate").click().run()                       # goes to AI Summary and generates
        assert not at.exception, at.exception
        assert at.session_state["nav"] == "AI Summary" and len(fake.requests) == 1
        go(at, "Home")
        text = page_text(at)
        assert "Laptops drive almost half of revenue" in text and "Laptops lead revenue" in text
        assert "Uncited advice" not in text                             # validation still applies
        assert any(b.key == "home_open_summary" for b in at.button)


def test_home_quick_navigation():
    at = run_with_sample()
    for page in ["Overview", "Charts", "AI Summary", "Ask AI", "Data"]:
        go(at, "Home")
        button(at, f"home_nav_{page}").click().run()
        assert not at.exception and at.session_state["nav"] == page


def test_new_dataset_opens_on_home():
    at = go(run_with_sample(), "Charts")
    at.selectbox(key="sample").set_value("Customer support tickets").run()
    assert at.session_state["nav"] == "Home"


# ---------------------------------------------------------------- guide and public-demo limits

def test_guide_from_welcome_and_back():
    at = AppTest.from_file(APP, default_timeout=90).run()
    button(at, "welcome_guide").click().run()
    assert not at.exception, at.exception
    text = page_text(at)
    assert "How to use BizInsight AI" in text and "Asking good questions".upper() in text.upper()
    assert "upload confidential or personal data" in text
    button(at, "guide_back").click().run()
    assert "Start in one click".upper() in page_text(at).upper()


def test_guide_in_navigation_after_loading_data():
    at = go(run_with_sample(), "Guide")
    assert "Understanding the numbers".upper() in page_text(at).upper()


def test_shared_daily_limit_applies_across_sessions(monkeypatch):
    monkeypatch.setenv("AI_DAILY_LIMIT", "1")
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(summary_json()))
        at = start_with_fake(monkeypatch, fake)
        button(at, "ai_generate").click().run()
        assert not at.exception, at.exception
        assert "shared daily limit" in page_text(at)

        # A brand-new visitor (new session) is blocked too, but the rest of the app still works.
        other = start_with_fake(monkeypatch, fake)
        assert button(other, "ai_generate").disabled
        assert "shared daily limit" in page_text(other)
        assert len(fake.requests) == 1


def test_summary_is_shared_with_later_visitors_of_the_same_file(monkeypatch):
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(summary_json()))
        first = start_with_fake(monkeypatch, fake)
        button(first, "ai_generate").click().run()
        assert "Laptops drive almost half of revenue" in page_text(first)

        second = start_with_fake(monkeypatch, fake)                    # a new visitor, same sample
        assert "Laptops drive almost half of revenue" in page_text(second)
        assert "10 of 10 AI requests left" in page_text(second)         # nothing spent
        assert len(fake.requests) == 1


def test_shared_answer_is_free_even_after_the_daily_limit(monkeypatch):
    monkeypatch.setenv("AI_DAILY_LIMIT", "2")                          # one question = one request
    with FakeOpenAI() as fake:
        fake.queue(200, response_body(plan_json(operation="aggregate", group_by=None)))
        fake.queue(200, response_body(answer_json("Total revenue is about 2.6 million.", ["Q1"],
                                                  headline="Total revenue is about 2.6 million")))
        fake.queue(200, response_body(plan_json()))
        fake.queue(200, response_body(answer_json("North sells the most.", ["Q1"])))
        first = start_with_fake(monkeypatch, fake, page="Ask AI")
        button(first, "example_What is the total Revenue?").click().run()
        first.chat_input[0].set_value("Which region sells the most?").run()
        assert len(fake.requests) == 4 and "shared daily limit" in page_text(first)

        second = start_with_fake(monkeypatch, fake, page="Ask AI")      # limit reached for everyone…
        cached = button(second, "example_What is the total Revenue?")
        assert not cached.disabled                                     # …but this answer is already known
        cached.click().run()
        assert "Total revenue is about 2.6 million." in page_text(second)
        assert len(fake.requests) == 4                                 # served without a new request
