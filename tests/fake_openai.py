"""A tiny fake OpenAI server for tests and demos — no API key, no cost, no internet.

It listens on localhost and answers POST /v1/responses with whatever the test
queued up. The real `openai` SDK talks to it exactly as it would to OpenAI
(we just point `base_url` at it), so the full request/response path is tested.

Run it by hand for a visual demo:
    python tests/fake_openai.py        # then start the app with
    OPENAI_API_KEY=sk-fake-key-for-demo OPENAI_BASE_URL=http://127.0.0.1:8765/v1 streamlit run app.py
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def response_body(text: str, status: str = "completed", incomplete_reason: str | None = None,
                  model: str = "gpt-5-mini") -> dict:
    """A minimal Responses-API JSON body containing `text` as the model's output."""
    return {
        "id": "resp_fake", "object": "response", "created_at": 0, "status": status, "model": model,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "output": [{"type": "message", "id": "msg_fake", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 1850, "output_tokens": 640, "total_tokens": 2490,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    }


def error_body(message: str, code: str, type_: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": type_, "code": code, "param": None}}


class FakeOpenAI:
    """Context manager: `with FakeOpenAI() as fake: fake.queue(200, body)`."""

    def __init__(self, port: int = 0):
        self.requests: list[dict] = []
        self._replies: list[tuple[int, dict]] = []
        # Optional function that builds a reply from the incoming request (used by the demo).
        self.responder = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (name required by http.server)
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(request)
                if fake._replies:
                    status, body = fake._replies.pop(0)
                elif fake.responder:
                    status, body = fake.responder(request)
                else:
                    status, body = 500, error_body("No reply queued in FakeOpenAI", "test_error")
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep test output quiet
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def queue(self, status: int, body: dict) -> None:
        self._replies.append((status, body))

    def __enter__(self) -> "FakeOpenAI":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def demo_summary(facts_text: str) -> dict:
    """Build a plausible summary that cites real fact IDs found in the request (for the demo server)."""
    ids = re.findall(r'"id": "(F\d+)"', facts_text) or ["F1"]
    pick = lambda i: [ids[min(i, len(ids) - 1)]]  # noqa: E731
    return {
        "headline": "Demo summary from the fake OpenAI server — not a real AI response.",
        "key_findings": [{"text": "This text comes from tests/fake_openai.py, not a real model.", "fact_ids": pick(0)}],
        "trends": [{"text": "Placeholder trend statement.", "fact_ids": pick(5)}],
        "data_quality": [{"text": "Placeholder data-quality statement.", "fact_ids": pick(len(ids) - 1)}],
        "recommendations": [{"text": "Placeholder recommendation.", "fact_ids": pick(2)}],
        "limitations": ["Demo mode: no real analysis was performed."],
    }


if __name__ == "__main__":
    import time

    server = FakeOpenAI(port=8765)
    server.responder = lambda req: (200, response_body(json.dumps(demo_summary(req.get("input", "")))))
    with server:
        print(f"Fake OpenAI server running at {server.base_url} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
