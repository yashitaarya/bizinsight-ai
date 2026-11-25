"""Stage 4 — a thin, safe wrapper around the OpenAI API.

Why a wrapper instead of calling `openai` everywhere?
  * One place reads the API key (from environment variables — never from code).
  * One place turns OpenAI's many error types into short, friendly messages.
  * Tests can swap in a fake client, so the test suite never needs a real key
    or costs money.

The key is read from the OPENAI_API_KEY environment variable. app.py fills the
environment from a local `.env` file (git-ignored) or Streamlit secrets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import openai

DEFAULT_MODEL = "gpt-5-mini"


class LLMError(Exception):
    """An AI request failed. `str(err)` is a message that is safe to show to users.

    `kind` lets the UI react differently, e.g. show setup help for "missing_key".
    """

    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


@dataclass
class LLMResult:
    data: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int


def get_settings(env: dict[str, str] | None = None) -> tuple[str | None, str]:
    """Return (api_key or None, model name) from environment variables."""
    env = os.environ if env is None else env
    key = (env.get("OPENAI_API_KEY") or "").strip() or None
    if key and key.startswith("sk-your-key"):
        key = None   # still the placeholder from .env.example
    model = (env.get("OPENAI_MODEL") or "").strip() or DEFAULT_MODEL
    return key, model


def mask_key(key: str | None) -> str:
    """'sk-proj-abc…wxyz' — enough to recognize a key without revealing it."""
    if not key:
        return "(not set)"
    return f"{key[:7]}…{key[-4:]}" if len(key) > 12 else "…"


def _friendly_error(err: Exception, model: str) -> LLMError:
    """Translate OpenAI SDK exceptions into plain-English messages."""
    if isinstance(err, openai.AuthenticationError):
        return LLMError("OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file "
                        "(it may be mistyped, revoked or expired).", "auth")
    if isinstance(err, openai.PermissionDeniedError):
        return LLMError(f"Your OpenAI key isn't allowed to use the model '{model}'. "
                        "Check your project's model permissions on platform.openai.com.", "auth")
    if isinstance(err, openai.NotFoundError):
        return LLMError(f"The model '{model}' isn't available to your account. "
                        "Set OPENAI_MODEL in .env to a model you can use.", "model")
    if isinstance(err, openai.RateLimitError):
        code = getattr(err, "code", None) or ""
        if "quota" in code or "quota" in str(err).lower() or "billing" in str(err).lower():
            return LLMError("Your OpenAI account has run out of credit or hit its spending limit. "
                            "Add credit or raise the limit under Billing on platform.openai.com.", "quota")
        return LLMError("OpenAI is receiving too many requests from this key right now. "
                        "Wait a minute and try again.", "rate_limit")
    if isinstance(err, openai.APITimeoutError):
        return LLMError("OpenAI took too long to respond. Please try again.", "timeout")
    if isinstance(err, openai.APIConnectionError):
        return LLMError("Couldn't reach OpenAI. Check your internet connection and try again.", "network")
    if isinstance(err, openai.BadRequestError):
        return LLMError(f"OpenAI couldn't process the request: {getattr(err, 'message', err)}", "bad_request")
    if isinstance(err, openai.APIStatusError) and err.status_code >= 500:
        return LLMError("OpenAI is having temporary problems. Please try again in a few minutes.", "server")
    return LLMError(f"Unexpected error while contacting OpenAI: {err}", "error")


class LLMClient:
    """Sends one request and returns validated JSON matching a schema."""

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, *,
                 timeout: float = 90, max_retries: int = 2, http_client=None, base_url: str | None = None):
        if not api_key:
            raise LLMError("No OpenAI API key is set. Add OPENAI_API_KEY to your .env file.", "missing_key")
        self.model = model
        # max_retries: the SDK automatically retries brief network/rate-limit hiccups.
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries,
                                     http_client=http_client, base_url=base_url)

    def generate_json(self, instructions: str, user_input: str, schema: dict, schema_name: str,
                      max_output_tokens: int = 6000) -> LLMResult:
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=user_input,
                # Structured Outputs: the model must reply with JSON matching `schema`.
                text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
                max_output_tokens=max_output_tokens,
                store=False,   # ask OpenAI not to keep this response on their side
            )
        except openai.OpenAIError as err:
            raise _friendly_error(err, self.model) from err

        for item in response.output or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "refusal":
                    raise LLMError("The AI declined to answer this request.", "refusal")
        if response.status != "completed":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None) or response.status
            if reason == "max_output_tokens":
                raise LLMError("The AI's answer was cut off because it was too long. Please try again.", "incomplete")
            raise LLMError(f"The AI response didn't complete ({reason}). Please try again.", "incomplete")
        try:
            data = json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as err:
            raise LLMError("The AI returned an answer in an unexpected format. Please try again.", "bad_output") from err

        usage = response.usage
        return LLMResult(
            data=data,
            model=response.model or self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
