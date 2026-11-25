"""How to use BizInsight AI — a short guide for first-time visitors.

Static content only: no data, no AI. Shown from the welcome screen and as the last page
in the navigation bar once a dataset is open.
"""

from __future__ import annotations

import streamlit as st

from core import config
from ui.components import callout, section_label


def _cards(items: list[tuple[str, str]], columns: int) -> None:
    for i in range(0, len(items), columns):
        for box, (title, body) in zip(st.columns(columns, gap="medium"), items[i:i + columns]):
            with box.container(border=True):
                st.markdown(f'<div class="bi-nav-title">{title}</div><div class="bi-nav-body">{body}</div>',
                            unsafe_allow_html=True)


def guide_page() -> None:
    st.markdown('<div class="bi-kicker">Guide</div><div class="bi-hero" style="font-size:2rem">'
                'How to use BizInsight AI</div><div class="bi-hero-sub">A two-minute tour: load some data, '
                'explore the dashboard, then let the AI explain it. Every number comes from a calculation '
                'you can inspect.</div>', unsafe_allow_html=True)

    section_label("Quick start")
    steps = [
        ("1", "Load data", "On the welcome screen, click <b>Explore retail sales sample</b> or "
                           "<b>Explore support tickets sample</b>, or upload your own CSV or Excel file in the "
                           f"sidebar (up to {config.MAX_FILE_MB} MB)."),
        ("2", "Explore", "The <b>Home</b> dashboard opens with key metrics, a performance chart and a data-health "
                         "check. Use the navigation bar at the top to switch pages."),
        ("3", "Ask the AI", "Open <b>AI Summary</b> and click <b>Generate summary</b>, or open <b>Ask AI</b> "
                            "and type a question such as <i>“Which product makes the most profit?”</i>"),
    ]
    for box, (n, title, body) in zip(st.columns(3, gap="medium"), steps):
        box.markdown(f'<div class="bi-step">STEP {n}</div><div class="bi-nav-title">{title}</div>'
                     f'<div class="bi-nav-body">{body}</div>', unsafe_allow_html=True)

    section_label("The pages")
    _cards([
        ("Home", "An at-a-glance dashboard. The metrics are chosen from your columns — revenue and orders for "
                 "sales data, response times for support data — and nothing is invented."),
        ("Overview", "Totals for your number columns, what each column contains, and a data-quality check "
                     "(missing values, duplicates, unusual values). Read this before trusting conclusions."),
        ("AI Summary", "An executive briefing: headline, key findings, trends, takeaways and recommendations, "
                       "each linked to the calculations behind it."),
        ("Ask AI", "Plain-English questions answered with KPI cards, a chart, a table and a business "
                   "takeaway. Suggested questions to try appear above the question box."),
        ("Charts", "Recommended trends, rankings and comparisons with date and category filters, plus a chart "
                   "builder. Hover over any chart for exact values."),
        ("Data", "A preview of the rows and a list of every automatic cleaning step, such as "
                 "“$1,200” becoming the number 1200."),
    ], columns=3)

    section_label("Asking good questions")
    left, right = st.columns(2, gap="medium")
    with left:
        callout("Questions that work well", [
            "Totals and averages: “What is the total revenue?”",
            "Rankings: “Which region has the highest average order value?”",
            "Trends: “How did revenue change by quarter in the North region?”",
            "Relationships: “Is there a relationship between unit price and revenue?”",
            "Counts: “How many orders were over 5,000?”",
            "Follow-ups: “What about the West?” (the previous question is remembered)",
        ], "info")
    with right:
        callout("What it won't do", [
            "Forecasts (“What will revenue be next year?”) — the data shows what happened, not what will.",
            "“Why” questions — the data shows patterns, not causes.",
            "Two analyses at once (“North in Q1 vs West in Q4”) — ask them as two questions.",
            "Answer from information that isn't in your file.",
            "Instead of guessing, it explains why and suggests questions it can answer.",
        ], "neutral")

    section_label("Understanding the numbers")
    _cards([
        ("pandas calculates, the AI explains", "Every figure is calculated by the app with Python and pandas. "
                                               "The AI only chooses an approved analysis and writes the "
                                               "explanation — it never does the math."),
        ("Sources & calculations", "Open this section under any summary or answer to see the exact facts and "
                                   "calculation it is based on. Answers also show how your question was "
                                   "understood."),
        ("Checked before you see it", "Statements without a supporting calculation are removed, and any figure "
                                      "that doesn't match a calculation is flagged on screen."),
    ], columns=3)

    section_label("Your data, privacy and limits")
    callout("Good to know", [
        "Uploaded files are processed in memory for your session and never saved. Refreshing the page clears "
        "them.",
        "AI features send OpenAI only column names, category labels and calculated results — never rows or the "
        "file itself.",
        "This is a public demo: please don't upload confidential or personal data. The two samples are "
        "synthetic.",
        f"New AI requests are limited to a small shared number per day ({config.AI_DAILY_LIMIT} by default) to "
        "protect the host's OpenAI credit. Summaries and answers that someone already generated for the same "
        "file — including the samples — are shown instantly and don't count. Everything except AI Summary and "
        "Ask AI works without limits.",
        "If the app has been idle, it may take a minute to wake up.",
    ], "neutral")
