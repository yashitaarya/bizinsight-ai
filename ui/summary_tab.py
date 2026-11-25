"""AI Summary tab: an executive briefing built on calculated facts.

Layout (top to bottom):
  headline + overview (AI)  ·  KPI cards (pandas)  ·  two key charts (pandas)
  key findings | business takeaways  ·  trends  ·  recommendations
  data quality (warnings) | limitations  ·  "Sources & calculations" (fact references, checks, usage)
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.charts import make_chart, recommend_charts
from core.data_loader import LoadedDataset
from core.facts import Fact
from core.formatting import Formatter, tidy_ai_text
from core.highlights import summary_highlights
from core.insights import SECTIONS, BusinessSummary, build_request, generate_summary
from core.llm_client import LLMClient, LLMError
from core.profiler import DatasetProfile
from ui.components import callout, empty_state, headline, insight_cards, key_charts, kpi_row, section_label
from ui.state import (Source, ai_settings, cached_facts, requests_left, setup_help, summary_key, usage_caption,
                      use_request)

SECTION_TITLES = {
    "key_findings": "Key findings",
    "trends": "Trends",
    "business_takeaways": "Business takeaways",
    "recommendations": "Recommendations",
    "data_quality": "Data quality",
}


def sources_section(summary: BusinessSummary, facts: list[Fact], generated: str) -> None:
    by_id = {f.id: f for f in facts}
    with st.expander("Sources & calculations"):
        st.caption("Every statement above cites facts calculated by BizInsight (pandas). The AI chose the wording; "
                   "it did not calculate any figures.")
        checks = [f"{summary.removed} statement(s) were removed because they didn't cite any calculated fact."
                  if summary.removed else "",
                  f"{summary.flagged} statement(s) contain a figure that couldn't be matched to the facts "
                  "(marked “Figure not verified”)." if summary.flagged else ""]
        callout("Verification", [c for c in checks if c], "warn")
        for name in SECTIONS:
            items = summary.section(name)
            if not items:
                continue
            st.markdown(f"**{SECTION_TITLES[name]}**")
            for item in items:
                refs = "; ".join(f"{i}: {by_id[i].text}" for i in item.fact_ids if i in by_id)
                st.markdown(f"- *{item.title}* — {refs}")
        st.markdown("**All calculated facts available to the AI**")
        st.dataframe(pd.DataFrame([f.to_dict() for f in facts]).rename(
            columns={"id": "ID", "category": "Type", "text": "Fact"}), hide_index=True, width="stretch")
        st.caption(f"Generated {generated} by {summary.model} · {summary.input_tokens:,} input + "
                   f"{summary.output_tokens:,} output tokens.")


def render_summary(summary: BusinessSummary, facts: list[Fact], generated: str) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        section_label(SECTION_TITLES["key_findings"])
        insight_cards(summary.key_findings)
    with right:
        section_label(SECTION_TITLES["business_takeaways"])
        if summary.business_takeaways:
            insight_cards(summary.business_takeaways)
        else:
            st.caption("No business takeaways were provided.")
    if summary.trends:
        section_label(SECTION_TITLES["trends"])
        insight_cards(summary.trends)
    if summary.recommendations:
        section_label(SECTION_TITLES["recommendations"])
        insight_cards(summary.recommendations, numbered=True)

    left, right = st.columns(2, gap="large")
    with left:
        callout("Data quality", [tidy_ai_text(f"{i.title}. {i.text}".replace("..", ".") if i.text else i.title)
                                 for i in summary.data_quality], "warn")
    with right:
        callout("Limitations of this analysis", [tidy_ai_text(x) for x in summary.limitations], "neutral")
    sources_section(summary, facts, generated)


def summary_tab(source: Source, ds: LoadedDataset, profile: DatasetProfile, fmt: Formatter) -> None:
    facts = cached_facts(*source)
    key, model = ai_settings()
    state = st.session_state.setdefault("ai_summaries", {})
    fingerprint = summary_key(source, model)
    entry = state.get(fingerprint)          # (BusinessSummary, generated-at text) or None

    top, action = st.columns([4, 1.3], vertical_alignment="bottom")
    with top:
        st.subheader("Executive summary")
        st.caption("An AI-written briefing based only on figures BizInsight has calculated. "
                   "It never sees your rows and doesn't calculate anything itself.")
    clicked = bool(st.session_state.pop("ai_autogenerate", False)) and entry is None   # from the Home page
    if key is not None:
        clicked = action.button("Regenerate" if entry else "Generate summary", type="secondary" if entry else "primary",
                                disabled=requests_left() <= 0, key="ai_generate", width="stretch") or clicked

    if entry:
        headline(tidy_ai_text(entry[0].headline) or "Executive summary", tidy_ai_text(entry[0].overview))
    kpi_row(summary_highlights(ds.df, profile, fmt))
    key_charts(ds.df, profile, key_prefix="summary_chart")

    if key is None:
        setup_help()
    elif clicked and requests_left() > 0:
        use_request()
        try:
            with st.spinner("Writing your executive summary… (usually 10–40 seconds)"):
                summary = generate_summary(LLMClient(key, model), facts, profile, source[0])
            state[fingerprint] = (summary, datetime.now().strftime("%Y-%m-%d %H:%M"))
            st.rerun()
        except LLMError as err:
            callout("The summary couldn't be generated", [str(err)], "warn")

    if key is not None and requests_left() <= 0:
        callout("AI request limit reached", ["This session's limit protects your OpenAI credit. "
                                             "Refresh the page to start a new session."], "warn")
    if entry:
        render_summary(*entry[:1], facts, entry[1])
    elif key is not None:
        empty_state("AI summary", "Generate an executive summary",
                    "Key findings, trends, business takeaways and recommendations — each linked to the "
                    "calculation it is based on.")

    if key is not None:
        st.caption(usage_caption(model))
    with st.expander(f"What is sent to OpenAI ({len(facts)} calculated facts, no rows)"):
        st.caption("Column names and types plus the calculated facts below — never rows, IDs or raw values. "
                   "Requests ask OpenAI not to store them.")
        st.code(build_request(facts, profile, source[0]), language="json")
