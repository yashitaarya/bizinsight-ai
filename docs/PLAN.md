# BizInsight AI — Project Plan

> **Historical document.** This is the plan written before development began, kept to show how the
> project was scoped. All stages are complete. For the current design see
> [ARCHITECTURE.md](ARCHITECTURE.md); the differences are summarized under [Outcome](#outcome) below.

An AI-powered Business Intelligence assistant: upload a CSV/Excel file, get an automatic
analysis, interactive charts, and answers to plain-English questions about the data.

## 1. Architecture

```
┌──────────────────────────── Streamlit app (app.py) ────────────────────────────┐
│  Sidebar: upload file / sample data / sheet picker                             │
│  Tabs:  Data  │  Overview  │  Charts  │  Ask your data  │  AI insights          │
└────────┬───────────┬───────────┬──────────────┬──────────────────┬─────────────┘
         │           │           │              │                  │
   data_loader   profiler     charts      query_engine        insights
   (read +       (stats,      (Plotly     (safe, whitelisted   (prompt built
   validate +    quality,     figures)    pandas operations)   from computed
   clean)        column                          ▲              facts only)
                 roles)                          │                  │
                                                 └──── llm_client ──┘
                                                  (OpenAI API, key from
                                                   env / st.secrets)
```

**Key design rule — "the AI never does the math."**
Pandas computes every number. The language model only (a) chooses which pre-approved
analysis to run, and (b) explains results that pandas already computed. This is how the
app avoids making unsupported claims:

1. User asks: "Which region had the highest revenue?"
2. The LLM returns a structured request, e.g. `{"operation": "group_aggregate", "group_by": "Region", "metric": "Revenue", "agg": "sum"}`.
3. Our code validates it (columns exist, operation is on the whitelist) and runs it with pandas.
4. The LLM writes the answer using **only** that result table, and the app shows the table + chart next to the answer so the user can verify it.

We deliberately do **not** let the AI write and execute arbitrary Python (`exec`/`eval`) —
that would let a malicious prompt or a crafted spreadsheet run code on your machine.

### Folder layout
```
bizinsight-ai/
├── app.py                  # Streamlit entry point (UI only)
├── core/
│   ├── config.py           # limits & settings (file size, model name)
│   ├── data_loader.py      # Stage 1: read/validate/clean uploads
│   ├── profiler.py         # Stage 2: stats, data quality, column roles
│   ├── charts.py           # Stage 3: Plotly chart builders
│   ├── llm_client.py       # Stage 4: thin OpenAI wrapper
│   ├── insights.py         # Stage 4: AI executive summary
│   └── query_engine.py     # Stage 5: safe natural-language Q&A
├── tests/                  # pytest unit tests
├── sample_data/            # synthetic demo dataset
├── .streamlit/config.toml  # theme + upload size limit
├── requirements.txt
├── .env.example            # template — real .env is git-ignored
└── README.md
```

## 2. Dependencies & setup

| Package | Why |
|---|---|
| streamlit | Web UI in pure Python |
| pandas | Data loading and analysis |
| openpyxl | Lets pandas read `.xlsx` |
| plotly | Interactive charts |
| openai | Official OpenAI SDK |
| python-dotenv | Loads the API key from a local `.env` file |
| pytest | Unit tests (dev only) |

Setup: Python 3.10+, a virtual environment, `pip install -r requirements.txt`, `streamlit run app.py`.

## 3. Stages (weekend plan)

| # | Stage | AI needed? | When |
|---|---|---|---|
| 1 | Scaffold, safe upload (CSV/XLSX), cleaning log, preview | No | Sat AM |
| 2 | Automatic profiling: KPIs, data-quality report, column stats | No | Sat PM |
| 3 | Interactive Plotly charts: auto-suggested + custom chart builder | No | Sat PM |
| 4 | OpenAI integration + AI executive summary & recommendations | Yes | Sun AM |
| 5 | Natural-language Q&A via the safe query engine | Yes | Sun PM |
| 6 | Polish, error handling, README, push to GitHub, optional deploy | No | Sun PM |

Stages 1–3 work fully without an API key, so the app is useful even offline.

## 4. API keys & external services

- **OpenAI API key** (required from Stage 4) — create at platform.openai.com, add prepaid credit.
  Stored in a local `.env` file (git-ignored) or Streamlit secrets; never in code.
- **GitHub account** (free) — version control and portfolio.
- **Streamlit Community Cloud** (optional, free) — public hosting; the key goes in its Secrets panel.

## 5. Estimated costs

- Streamlit, pandas, Plotly, GitHub, Streamlit Community Cloud: **free**.
- OpenAI: pay per token. We send summaries of the data (column names, stats, small
  result tables), never the full file, so each request is small (~2–4k tokens in, ~0.5k out).
  With a small model such as `gpt-5-mini` (~$0.25 / 1M input, ~$2 / 1M output tokens as
  of late 2026), that's roughly **$0.001–0.003 per question**. A full weekend of
  development and testing should cost **well under $2**; a $5 credit lasts a long time.
  The model name is configurable in `.env`, and prices change — check OpenAI's pricing page.

## 6. Data safety

- Uploads are kept in memory for the session only; nothing is written to disk.
- File type and size limits (25 MB) enforced both by Streamlit and our own code.
- Only aggregated statistics and small result tables are sent to OpenAI — never the full dataset.
  The UI states this clearly before any AI call.
- No `eval`/`exec` of model output; the AI can only pick from whitelisted operations.
- Every automatic cleaning step (e.g. "$1,200" → 1200) is logged and shown to the user.

## Outcome

All six stages were delivered. Compared with the original plan, the final build:

- **Added a fact sheet** (`core/facts.py`): instead of sending "summary statistics", pandas produces
  numbered facts that the AI must cite, which made automatic citation and number checks possible.
- **Split Q&A into its own module** (`core/qa.py`) with a two-call flow (plan, then explain) around the
  validated pandas query engine, plus a structured "can't answer" path with suggested questions.
- **Moved the interface into a `ui/` package** with shared components, and replaced tabs with a
  navigation bar so the new **Home dashboard** can link to every page.
- **Added a presentation layer**: consistent formatting (`core/formatting.py`), currency detection from
  the data, and pandas-calculated KPI cards and dashboard metrics (`core/highlights.py`).
- **Added a second synthetic dataset** (customer support tickets) to prove the app adapts to non-sales data.
- **Grew the test suite** to 231 test cases, including Streamlit `AppTest` tests of the full app and a local
  fake OpenAI server, plus an opt-in live check (`scripts/live_check.py`).
- Cost estimates held: each AI request sends a few thousand tokens of facts and typically costs a fraction
  of a cent with a small model.
