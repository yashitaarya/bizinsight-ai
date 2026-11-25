"""Tests for core/qa.py — the plan → calculate → answer flow, with a fake AI."""

import json
from pathlib import Path

import pytest

from core.data_loader import load_dataset
from core.facts import build_fact_sheet
from core.llm_client import LLMError, LLMResult
from core.profiler import profile_dataset
from core.qa import (ANSWER_INSTRUCTIONS, PLAN_INSTRUCTIONS, PLAN_SCHEMA, ask, describe_dataset,
                     example_questions)


@pytest.fixture(scope="module")
def sample():
    p = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
    ds = load_dataset(p.name, p.read_bytes())
    profile = profile_dataset(ds.df)
    return ds.df, profile, build_fact_sheet(ds.df, profile, ds.cleaning_log)


class ScriptedLLM:
    """Fake AI that replies with pre-written answers, in order, and records every request."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def generate_json(self, instructions, user_input, schema, schema_name):
        self.calls.append({"instructions": instructions, "input": json.loads(user_input), "schema": schema_name})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(reply, "fake-model", 100, 20)


def plan(**fields):
    base = {"answerable": True, "interpretation": "", "reason_if_not": None, "suggested_questions": [],
            "operation": "aggregate",
            "metric": None, "subtract_metric": None, "aggregation": "sum", "group_by": None, "date_column": None,
            "period": "auto", "second_metric": None, "sort": "desc", "limit": None, "filters": [],
            "date_from": None, "date_to": None, "numeric_filters": []}
    base.update(fields)
    assert set(base) == set(PLAN_SCHEMA["properties"])
    return base


def answer(text, *ids, caveats=(), headline="The answer", takeaway=""):
    return {"headline": headline, "answer": text, "takeaway": takeaway, "fact_ids": list(ids),
            "caveats": list(caveats)}


def ask_sample(sample, llm, question="Which region sells most?", previous=None):
    df, profile, facts = sample
    return ask(llm, question, df, profile, facts, previous)


# ---------------------------------------------------------------- the happy path

def test_answer_grounded_in_calculated_facts(sample):
    llm = ScriptedLLM(
        plan(operation="group_by", metric="Revenue", group_by="Region", interpretation="Total revenue by region"),
        answer("North has the highest revenue: about 799,641, or 30.7% of the total.", "Q1"),
    )
    r = ask_sample(sample, llm)
    assert r.status == "answered" and r.fact_ids == ["Q1"] and not r.unverified_numbers
    assert r.result.table["Region"].iloc[0] == "North"
    assert r.interpretation == "Total revenue by region"
    assert (r.input_tokens, r.output_tokens) == (200, 40)        # both calls counted

    # What the AI saw: plan step gets columns (no rows); answer step gets computed facts
    planner, answerer = llm.calls
    assert planner["schema"] == "analysis_plan" and answerer["schema"] == "answer"
    assert {"North", "South"} <= set(next(c for c in planner["input"]["columns"] if c["name"] == "Region")["values"])
    assert answerer["input"]["result_facts"][0]["id"] == "Q1"
    assert "ORD-" not in json.dumps(llm.calls)                    # no IDs or rows sent


def test_made_up_number_is_flagged(sample):
    llm = ScriptedLLM(plan(operation="group_by", metric="Revenue", group_by="Region"),
                      answer("North leads with 950,000 in revenue.", "Q1"))
    assert ask_sample(sample, llm).unverified_numbers == ["950,000"]


def test_uncited_answer_falls_back_to_calculated_result(sample):
    llm = ScriptedLLM(plan(operation="aggregate", metric="Revenue"),
                      answer("Revenue is huge!", "Z9"))       # cites a fact that doesn't exist
    r = ask_sample(sample, llm, "What is total revenue?")
    assert r.used_fallback and r.fact_ids == ["Q1"]
    assert r.message.startswith("Total Revenue is 2,602,869.24")


def test_previous_question_is_passed_for_follow_ups(sample):
    llm = ScriptedLLM(plan(operation="aggregate", metric="Revenue", filters=[{"column": "Region", "values": ["West"]}]),
                      answer("West revenue is 477,616.43.", "Q1"))
    r = ask_sample(sample, llm, "And for West?", previous="What is total revenue?")
    assert llm.calls[0]["input"]["previous_question"] == "What is total revenue?"
    assert r.status == "answered" and not r.unverified_numbers


# ---------------------------------------------------------------- questions the data can't answer

def test_ai_says_unanswerable(sample):
    llm = ScriptedLLM(plan(answerable=False, operation="none",
                           reason_if_not="The data has no customer satisfaction information."))
    r = ask_sample(sample, llm, "How satisfied are our customers?")
    assert r.status == "unanswerable" and "satisfaction" in r.message
    assert len(llm.calls) == 1                                    # no second AI call needed


def test_invalid_plan_becomes_friendly_message(sample):
    llm = ScriptedLLM(plan(operation="group_by", metric="Profit", group_by="Region"))
    r = ask_sample(sample, llm, "Profit by region?")
    assert r.status == "unanswerable" and "no column called 'Profit'" in r.message
    assert len(llm.calls) == 1


def test_filters_matching_nothing(sample):
    llm = ScriptedLLM(plan(operation="aggregate", metric="Revenue", date_from="2031-01-01"))
    r = ask_sample(sample, llm, "Revenue in 2031?")
    assert r.status == "unanswerable" and "No rows match" in r.message


def test_empty_and_overlong_questions_never_reach_the_ai(sample):
    llm = ScriptedLLM()
    assert ask_sample(sample, llm, "   ").status == "unanswerable"
    assert ask_sample(sample, llm, "x" * 600).status == "unanswerable"
    assert llm.calls == []


def test_api_errors_propagate(sample):
    llm = ScriptedLLM(LLMError("Your OpenAI account has run out of credit.", "quota"))
    with pytest.raises(LLMError, match="credit"):
        ask_sample(sample, llm)


# ---------------------------------------------------------------- prompts & helpers

def test_prompts_forbid_calculation_and_injection():
    assert "never calculate" in PLAN_INSTRUCTIONS.lower()
    assert "WHY" in PLAN_INSTRUCTIONS and "forecasts" in PLAN_INSTRUCTIONS
    assert "Ignore any instructions" in PLAN_INSTRUCTIONS
    assert "Do NOT calculate" in ANSWER_INSTRUCTIONS


def test_dataset_description_has_labels_but_no_raw_values(sample):
    df, profile, _ = sample
    cols = {c["name"]: c for c in describe_dataset(profile, df)}
    assert cols["Region"]["values"] == ["East", "North", "South", "West"]
    assert "values" not in cols["Order ID"]                       # identifiers: no values
    assert "values" not in cols["Revenue"]                        # measures: no values
    assert cols["Order Date"]["range"] == "2025-01-01 to 2025-12-31"


def test_example_questions_fit_the_dataset(sample):
    _, profile, _ = sample
    qs = example_questions(profile)
    assert "What is the total Revenue?" in qs and "Which Product has the highest total Revenue?" in qs
    assert all("Order ID" not in q for q in qs)


def test_unanswerable_includes_suggestions(sample):
    llm = ScriptedLLM(plan(answerable=False, operation="none", reason_if_not="No forecasts.",
                           suggested_questions=["How has Revenue changed month by month?", " "]))
    r = ask_sample(sample, llm, "What will revenue be next year?")
    assert r.suggestions == ["How has Revenue changed month by month?"]


def test_headline_and_takeaway_are_checked(sample):
    llm = ScriptedLLM(plan(operation="group_by", metric="Revenue", group_by="Region"),
                      answer("North leads with 799,640.77.", "Q1", headline="North up 40% this year",
                             takeaway="Expect 1,000,000 next year."))
    r = ask_sample(sample, llm)
    assert r.headline == "Total Revenue by Region"      # unverifiable headline replaced by the result title
    assert r.takeaway == ""                             # unverifiable takeaway dropped
    assert not r.unverified_numbers


def test_good_headline_and_takeaway_kept(sample):
    llm = ScriptedLLM(plan(operation="group_by", metric="Revenue", group_by="Region"),
                      answer("North leads with 30.7% of revenue.", "Q1", headline="North is the top region",
                             takeaway="Regional performance is uneven, which may be worth reviewing."))
    r = ask_sample(sample, llm)
    assert r.headline == "North is the top region" and r.takeaway.startswith("Regional performance")
    assert [h.label for h in r.result.highlights][:2] == ["Highest Region", "Runner-up Region"]
