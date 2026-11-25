"""Tests for core/llm_client.py against a local fake OpenAI server (no key, no cost)."""

import pytest

from core.llm_client import DEFAULT_MODEL, LLMClient, LLMError, get_settings, mask_key
from tests.fake_openai import FakeOpenAI, error_body, response_body


@pytest.fixture
def fake():
    with FakeOpenAI() as server:
        yield server


def client(fake, model="gpt-5-mini"):
    return LLMClient("sk-test-0123456789abcdef", model, base_url=fake.base_url, max_retries=0, timeout=10)


# ---------------------------------------------------------------- settings

def test_settings_from_environment():
    assert get_settings({"OPENAI_API_KEY": " sk-abc ", "OPENAI_MODEL": "gpt-x"}) == ("sk-abc", "gpt-x")
    assert get_settings({}) == (None, DEFAULT_MODEL)
    assert get_settings({"OPENAI_API_KEY": ""})[0] is None
    assert get_settings({"OPENAI_API_KEY": "sk-your-key-here"})[0] is None   # template placeholder


def test_mask_key_hides_the_secret():
    masked = mask_key("sk-proj-SECRETSECRETSECRET1234")
    assert "SECRETSECRET" not in masked and masked.endswith("1234")
    assert mask_key(None) == "(not set)"


def test_missing_key_is_a_friendly_error():
    with pytest.raises(LLMError) as err:
        LLMClient(None)
    assert err.value.kind == "missing_key"


# ---------------------------------------------------------------- happy path

def test_successful_structured_response(fake):
    fake.queue(200, response_body('{"headline": "Hi"}'))
    result = client(fake).generate_json("be careful", "facts here", {"type": "object"}, "summary")
    assert result.data == {"headline": "Hi"}
    assert (result.input_tokens, result.output_tokens) == (1850, 640)

    sent = fake.requests[-1]
    assert sent["model"] == "gpt-5-mini"
    assert sent["instructions"] == "be careful" and sent["input"] == "facts here"
    assert sent["store"] is False                                   # don't keep our data at OpenAI
    assert sent["text"]["format"] == {"type": "json_schema", "name": "summary",
                                      "schema": {"type": "object"}, "strict": True}
    assert sent["reasoning"] == {"effort": "low"}                   # cost control: little hidden reasoning
    assert sent["max_output_tokens"] == 2000                        # default ceiling on billed output


# ---------------------------------------------------------------- cost controls

def test_no_automatic_retries_by_default():
    llm = LLMClient("sk-test-0123456789abcdef")
    assert llm._client.max_retries == 0          # a retried request could be billed twice


def test_oversized_request_is_refused_before_sending(fake):
    from core import config
    with pytest.raises(LLMError) as err:
        client(fake).generate_json("x", "y" * (config.AI_MAX_INPUT_CHARS + 1), {"type": "object"}, "summary")
    assert err.value.kind == "too_large" and fake.requests == []    # nothing sent, nothing billed


def test_reasoning_effort_only_sent_to_reasoning_models(fake):
    fake.queue(200, response_body('{"a": 1}'))
    client(fake, model="gpt-4.1-mini").generate_json("i", "u", {"type": "object"}, "s", max_output_tokens=123)
    sent = fake.requests[-1]
    assert "reasoning" not in sent and sent["max_output_tokens"] == 123


# ---------------------------------------------------------------- failures

@pytest.mark.parametrize("status, body, kind, words", [
    (401, error_body("Incorrect API key", "invalid_api_key"), "auth", "rejected the API key"),
    (403, error_body("No access", "model_not_allowed"), "auth", "isn't allowed"),
    (404, error_body("Model not found", "model_not_found"), "model", "OPENAI_MODEL"),
    (429, error_body("You exceeded your current quota", "insufficient_quota", "insufficient_quota"),
     "quota", "run out of credit"),
    (429, error_body("Project spend limit reached", "project_spend_limit_exceeded", "requests"), "quota", "AI features are paused"),
    (429, error_body("Organization spend limit reached", "organization_spend_limit_exceeded", "requests"),
     "quota", "spending limit"),
    (429, error_body("No credit", "credit_balance_exhausted", "insufficient_quota"), "quota", "run out of credit"),
    (429, error_body("Rate limit reached", "rate_limit_exceeded", "requests"), "rate_limit", "Wait a minute"),
    (500, error_body("Server error", "server_error", "server_error"), "server", "temporary problems"),
    (400, error_body("Invalid schema", "invalid_schema"), "bad_request", "couldn't process"),
])
def test_api_errors_become_friendly_messages(fake, status, body, kind, words):
    fake.queue(status, body)
    with pytest.raises(LLMError) as err:
        client(fake).generate_json("i", "x", {}, "s")
    assert err.value.kind == kind and words in str(err.value)
    assert "sk-test" not in str(err.value)                          # never echo the key


def test_cut_off_answer(fake):
    fake.queue(200, response_body('{"head', status="incomplete", incomplete_reason="max_output_tokens"))
    with pytest.raises(LLMError, match="cut off"):
        client(fake).generate_json("i", "x", {}, "s")


def test_non_json_answer(fake):
    fake.queue(200, response_body("Sorry, here is some prose instead"))
    with pytest.raises(LLMError, match="unexpected format"):
        client(fake).generate_json("i", "x", {}, "s")


def test_refusal(fake):
    body = response_body("")
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "I can't help with that."}]
    fake.queue(200, body)
    with pytest.raises(LLMError) as err:
        client(fake).generate_json("i", "x", {}, "s")
    assert err.value.kind == "refusal"


def test_unreachable_server():
    llm = LLMClient("sk-test-0123456789abcdef", base_url="http://127.0.0.1:9/v1", max_retries=0, timeout=3)
    with pytest.raises(LLMError) as err:
        llm.generate_json("i", "x", {}, "s")
    assert err.value.kind in ("network", "timeout")
