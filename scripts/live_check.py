"""End-to-end check of the live OpenAI integration, using the bundled sample datasets.

Run from the project folder (with your virtual environment active):

    python scripts/live_check.py

What it does
  * Reads OPENAI_API_KEY / OPENAI_MODEL from your .env file (the key is never printed).
  * Generates an AI summary for each sample dataset and asks a few questions,
    including one the data cannot answer (a forecast).
  * Prints a pass/fail report: did the AI respond, did its statements cite real
    calculated facts, were any figures flagged as unverifiable?
  * Saves the AI's responses (not your key) to live_check_responses.json, which is
    git-ignored. These can be replayed to take README screenshots of real output.

Cost: about 12 small API requests — typically a few US cents with a "mini" model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from core.data_loader import load_dataset  # noqa: E402
from core.facts import build_fact_sheet  # noqa: E402
from core.formatting import Formatter  # noqa: E402
from core.insights import SECTIONS, generate_summary  # noqa: E402
from core.llm_client import LLMClient, LLMError, get_settings, mask_key  # noqa: E402
from core.profiler import profile_dataset  # noqa: E402
from core.qa import ask  # noqa: E402

OUTPUT = ROOT / "live_check_responses.json"

CHECKS = {
    "sales_sample.csv": [
        ("Which product makes the most profit?", "answered"),
        ("How did revenue change by quarter in the North region?", "answered"),
        ("What will revenue be next year?", "unanswerable"),
    ],
    "support_tickets.csv": [
        ("Which channel gets the fastest first response?", "answered"),
        ("Does satisfaction drop when resolution takes longer?", "answered"),
    ],
}


class RecordingClient(LLMClient):
    """An LLMClient that also keeps a copy of every structured response it receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict] = []
        self.context: dict = {}

    def generate_json(self, instructions, user_input, schema, schema_name, max_output_tokens=6000):
        result = super().generate_json(instructions, user_input, schema, schema_name, max_output_tokens)
        self.records.append({**self.context, "kind": schema_name, "data": result.data})
        return result


def main() -> int:
    load_dotenv(ROOT / ".env")
    key, model = get_settings()
    if key is None:
        print("No OPENAI_API_KEY found. Copy .env.example to .env and add your key first.")
        return 1

    print(f"Model: {model}   Key: {mask_key(key)}")
    print(f"This makes about {sum(len(q) * 2 + 1 for q in CHECKS.values())} API requests "
          "(typically a few US cents).")
    if "--yes" not in sys.argv and input("Continue? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    client = RecordingClient(key, model)
    failures, tokens = 0, 0

    for name, questions in CHECKS.items():
        path = ROOT / "sample_data" / name
        ds = load_dataset(path.name, path.read_bytes())
        profile = profile_dataset(ds.df)
        facts = build_fact_sheet(ds.df, profile, ds.cleaning_log)
        fmt = Formatter(ds.currency_symbol)
        print(f"\n=== {name} ===")

        client.context = {"dataset": name, "question": None}
        try:
            s = generate_summary(client, facts, profile, name)
            items = sum(len(s.section(sec)) for sec in SECTIONS)
            tokens += s.input_tokens + s.output_tokens
            print(f"[PASS] Summary: {s.headline!r}")
            print(f"       {items} statements kept · {s.removed} removed (no valid source) · "
                  f"{s.flagged} with unverified figures")
        except LLMError as err:
            failures += 1
            print(f"[FAIL] Summary: {err}")
            continue

        for question, expected in questions:
            client.context = {"dataset": name, "question": question}
            try:
                r = ask(client, question, ds.df, profile, facts, None, fmt)
            except LLMError as err:
                failures += 1
                print(f"[FAIL] {question}\n       {err}")
                continue
            tokens += r.input_tokens + r.output_tokens
            ok = r.status == expected
            failures += 0 if ok else 1
            print(f"[{'PASS' if ok else 'CHECK'}] {question}  →  {r.status} (expected {expected})")
            if r.status == "answered":
                print(f"       {r.headline}: {r.message}")
                flags = f" · unverified: {', '.join(r.unverified_numbers)}" if r.unverified_numbers else ""
                print(f"       cites {', '.join(r.fact_ids)}{flags}{' · used fallback' if r.used_fallback else ''}")
            else:
                print(f"       {r.message}")

    OUTPUT.write_text(json.dumps({"model": model, "records": client.records}, indent=1, ensure_ascii=False))
    print(f"\nTokens used: {tokens:,}.  Responses saved to {OUTPUT.name} (git-ignored; contains no key).")
    print("All checks passed." if failures == 0 else f"{failures} check(s) need attention — see above.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
