# BizInsight AI — guidelines for AI coding assistants

This file is read automatically by AI coding assistants (such as Claude Code) working in this
repository. Human contributors may find it a useful summary of the project's conventions.
See `README.md` for the product overview and `docs/ARCHITECTURE.md` for the full design.

## Working style
- Prefer small, reviewable changes; explain important decisions in plain language.
- Never claim something works without running it: run `pytest` and launch the app.
- Keep the app working and polished over adding complexity.
- Never initialize Git, commit, push or publish without the maintainer's explicit approval.

## Non-negotiable rules
- **The AI never does the math.** pandas computes every number. The model only picks an approved
  operation (validated against real columns) and explains calculated results.
- Never `eval`/`exec` model output; plans are data, validated by `core/query_engine.validate_plan`.
- Send OpenAI only column names/types, category labels and calculated facts — never rows or raw values.
- Read the API key from `.env` (python-dotenv) or `st.secrets`. Never hardcode, print or commit keys.
- Uploads stay in memory; never write user data to disk.
- `core/` must not import Streamlit (keeps it unit-testable). UI lives in `app.py` and `ui/`.
- Escape all data or AI text placed in HTML (`ui/components.py` uses `html.escape`).
- KPI cards come from `core/highlights.py` (pandas), never from the AI. Home "AI highlights" only show
  a summary the user generated and that passed validation.
- Format numbers via `core/formatting.Formatter`; show a currency symbol only if the data contained one.
- Show changes neutrally (▲/▼), never green/red — whether "up" is good depends on the metric.
- Display AI text through `tidy_ai_text` (and `md_safe` in Markdown) only after validation has run.
- Log every automatic data change in `LoadedDataset.cleaning_log`.
- Use US English spelling in code, UI text and documentation.

## Architecture notes
- AI summary: pandas builds numbered facts (F1, F2…) → OpenAI with a strict JSON schema
  (`store=False`) → statements citing no valid fact are dropped; numbers that don't match a fact
  (rounding allowed, % only matches %) are flagged.
- Q&A: plan (model) → `validate_plan` → `run_query` (pandas, result facts Q1…) → explanation (model) →
  same checks. Each question is two API calls and counts as one toward the session limit.
- Approved operations: `aggregate`, `group_by`, `time_trend`, `correlation`, `count_distinct`, plus
  category filters, date range, numeric conditions and optional metric-minus-metric. Adding an operation
  means updating `OPERATIONS`, `validate_plan`, `run_query`, `PLAN_SCHEMA`/`PLAN_INSTRUCTIONS` and tests.
- Charts: `recommend_charts(profile)` → `ChartSpec` → `make_chart(spec, df)` → figure + table + notes.
  Fixed color per category, one y-axis, totals start at zero, honest notes (cut-offs, partial periods).
- Navigation: `st.session_state["nav"]` with `ui/state.go_to()` callbacks; pages in `ui/state.PAGES`.
- AI requests per session are capped by `config.MAX_AI_REQUESTS_PER_SESSION` (10).

## Testing
- Tests never call OpenAI: `tests/fake_openai.py` is a local fake server, and `tests/test_app.py`
  blanks `OPENAI_API_KEY` and points `OPENAI_BASE_URL` nowhere unless a test starts the fake server.
- `tests/test_app.py` runs the real app headlessly with Streamlit's `AppTest`.
- `scripts/live_check.py` is the opt-in check against the real API (costs a few cents).

## Commands
```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                         # run the test suite
streamlit run app.py           # http://localhost:8501
python scripts/live_check.py   # optional: real OpenAI end-to-end check
```

## Environment notes
- Developed against Python 3.13, pandas 3.x, Streamlit 1.6x, OpenAI SDK 3.x.
- Use `width="stretch"`, not the deprecated `use_container_width`.
- The default model is `gpt-5-mini` (`OPENAI_MODEL`); reasoning models reject `temperature`, so don't pass it.
