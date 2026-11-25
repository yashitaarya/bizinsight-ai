"""Ask tab: plain-English questions answered from calculated results.

Each answer is laid out like a small report:
  headline · direct answer · KPI cards · chart + table · business takeaway · notes
  "Sources & calculations" (how the question was read, filters, cited facts)
Unanswerable questions get a clear explanation and alternative questions to try.
"""

from __future__ import annotations

import hashlib
import json

import streamlit as st

from core.charts import make_chart
from core.data_loader import LoadedDataset
from core.facts import Fact
from core.formatting import Formatter, tidy_ai_text
from core.llm_client import LLMClient, LLMError
from core.profiler import DatasetProfile
from core.qa import QAResponse, ask, describe_dataset, example_questions
from ui.components import callout, headline, kpi_row, md_safe, number_columns, section_label
from ui.state import (Source, ai_settings, cached_facts, limit_message, requests_left, setup_help, shared_get,
                      shared_put, usage_caption, use_request)

MAX_HISTORY = 20
USER_AVATAR, ASSISTANT_AVATAR = ":material/person:", ":material/insights:"


def _answer_key(fingerprint: str, model: str, question: str, previous: str | None) -> str:
    """Identifies an answer that can be reused: same file, model, question and previous question."""
    norm = lambda s: " ".join((s or "").lower().split())
    return f"qa|{fingerprint}|{model}|{norm(question)}|{norm(previous)}"


def _ask_button(question: str, key: str, cached: bool = False) -> None:
    if st.button(question, key=key, width="stretch", disabled=requests_left() <= 0 and not cached):
        st.session_state["qa_pending"] = question
        st.rerun()


def render_unanswerable(resp: QAResponse, examples: list[str], index: int) -> None:
    st.markdown(
        '<div class="bi-empty"><div class="kicker">Not answerable from this data</div>'
        '<div class="title">This question can\'t be answered from your dataset</div></div>',
        unsafe_allow_html=True)
    st.markdown(md_safe(tidy_ai_text(resp.message)))
    if resp.interpretation:
        st.caption(md_safe(f"How the question was understood: {tidy_ai_text(resp.interpretation)}"))
    alternatives = list(dict.fromkeys(resp.suggestions + examples))[:3]
    if alternatives:
        st.caption("Try one of these instead")
        for j, q in enumerate(alternatives):
            _ask_button(q, key=f"alt_{index}_{j}")


def render_answer(resp: QAResponse, all_facts: dict[str, Fact], fmt: Formatter, index: int) -> None:
    result = resp.result
    ids = {f.id for f in result.facts}
    headline(tidy_ai_text(resp.headline, ids) or result.title, small=True)
    st.markdown(md_safe(tidy_ai_text(resp.message, ids)))
    kpi_row(result.highlights)

    money = fmt.money_columns(result.table.columns)
    table_config = number_columns(result.table, money, fmt.currency)
    chart = None
    if result.chart is not None and result.frame is not None:
        try:
            chart = make_chart(result.chart, result.frame).figure
        except ValueError:
            chart = None
    if chart is not None:
        tab_chart, tab_table = st.tabs(["Chart", "Table"])
        with tab_chart:
            if result.chart.reason:
                st.caption(result.chart.reason)
            chart.update_layout(height=340)
            st.plotly_chart(chart, width="stretch", theme=None, key=f"qa_chart_{index}", config={"displaylogo": False})
        with tab_table:
            st.dataframe(result.table, hide_index=True, width="stretch", column_config=table_config)
    elif len(result.table) > 1:
        st.dataframe(result.table, hide_index=True, width="stretch", column_config=table_config)

    if resp.takeaway:
        callout("Business takeaway", [tidy_ai_text(resp.takeaway, ids)], "info")

    notes = [tidy_ai_text(c, ids) for c in resp.caveats] + result.notes
    if resp.unverified_numbers:
        notes.append(f"Could not match {', '.join(resp.unverified_numbers)} to the calculated results — "
                     "check the table before relying on it.")
    if resp.used_fallback:
        notes.append("The AI's wording didn't pass the checks, so this answer is taken directly from the calculation.")
    callout("Notes", notes, "neutral")

    with st.expander("Sources & calculations"):
        if resp.interpretation:
            st.markdown(md_safe(f"**How the question was understood:** {resp.interpretation}"))
        st.markdown(f"**Calculation:** {result.title}" + (f" — where {result.filter_text}" if result.filter_text else ""))
        st.markdown("**Facts cited in the answer**")
        for i in resp.fact_ids:
            if i in all_facts:
                st.markdown(f"- {i}: {all_facts[i].text}")
        st.markdown("**All calculated results for this question**")
        for f in result.facts:
            st.markdown(f"- {f.id}: {f.text}")
        st.caption(f"{resp.input_tokens:,} input + {resp.output_tokens:,} output tokens across two AI requests "
                   "(plan, then explanation).")


def render_turn(resp: QAResponse, dataset_facts: dict[str, Fact], fmt: Formatter, examples: list[str],
                index: int) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(resp.question)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if resp.status == "error":
            callout("Something went wrong", [resp.message], "warn")
        elif resp.status == "unanswerable":
            render_unanswerable(resp, examples, index)
        else:
            facts = dict(dataset_facts)
            facts.update({f.id: f for f in resp.result.facts})
            render_answer(resp, facts, fmt, index)


def ask_tab(source: Source, ds: LoadedDataset, profile: DatasetProfile, fmt: Formatter) -> None:
    st.subheader("Ask your data")
    st.caption("Ask a business question in plain English. The AI picks one of BizInsight's approved analyses, "
               "pandas calculates the result, and the AI explains it.")

    key, model = ai_settings()
    dataset_facts = cached_facts(*source)
    examples = example_questions(profile)

    if key is None:
        setup_help()
    else:
        fingerprint = hashlib.sha256(source[1]).hexdigest()[:16] + f"|{source[2]}"
        history: list[QAResponse] = st.session_state.setdefault("qa_history", {}).setdefault(fingerprint, [])

        if examples:
            section_label("Suggested questions")
            grid = st.columns(2)
            last = next((h.question for h in reversed(history) if h.status == "answered"), None)
            for i, q in enumerate(examples):
                with grid[i % 2]:
                    _ask_button(q, key=f"example_{q}",
                                cached=shared_get(_answer_key(fingerprint, model, q, last)) is not None)

        facts_by_id = {f.id: f for f in dataset_facts}
        for i, resp in enumerate(history):
            render_turn(resp, facts_by_id, fmt, examples, i)

        typed = st.chat_input("Ask a question about your data…", key="qa_input", max_chars=500,
                              disabled=requests_left() <= 0)
        left, right = st.columns([3, 1], vertical_alignment="center")
        left.caption(usage_caption(model) + " · each question uses one")
        if history and right.button("Clear conversation", key="qa_clear", width="stretch"):
            history.clear()
            st.rerun()
        if requests_left() <= 0:
            callout("AI request limit reached", [limit_message()], "warn")

        question = typed or st.session_state.pop("qa_pending", None)
        if question:
            previous = next((h.question for h in reversed(history) if h.status == "answered"), None)
            answer_key = _answer_key(fingerprint, model, question, previous)
            resp = shared_get(answer_key)            # already answered for this exact file: free
            if resp is None and requests_left() > 0:
                use_request()
                try:
                    with st.spinner("Working out how to answer…"):
                        resp = ask(LLMClient(key, model), question, ds.df, profile, dataset_facts, previous, fmt)
                except LLMError as err:
                    resp = QAResponse(question, "error", str(err))
                if resp.status != "error":
                    shared_put(answer_key, resp)
            if resp is not None:
                history.append(resp)
                del history[:-MAX_HISTORY]
                st.rerun()

    with st.expander("What is sent to OpenAI for each question"):
        st.markdown("1. **Your question**, plus column names, column types and the labels of category columns "
                    "(e.g. region names), so the AI can choose the right analysis.\n"
                    "2. **The calculated results** for your question and the dataset's calculated facts, so it can "
                    "explain them.\n\nNever sent: rows, IDs, free-text values or the file itself.")
        st.code(json.dumps(describe_dataset(profile, ds.df), indent=1, ensure_ascii=False), language="json")
