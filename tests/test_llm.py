import json

import httpx

from musicteacher.llm.base import LLMResponse, Message
from musicteacher.llm.cache import CachedLLMClient, cache_key
from musicteacher.llm.fake import FakeLLMClient
from musicteacher.llm.ollama_client import OllamaClient
from musicteacher.llm.usage import UsageTracker

QUESTION = [Message("user", "Notes of a C major triad?")]


def test_cache_hit_does_not_call_the_model(tmp_path):
    fake = FakeLLMClient(replies=["C E G"])
    client = CachedLLMClient(fake, tmp_path)

    first = client.complete(QUESTION)
    second = client.complete(QUESTION)

    assert len(fake.calls) == 1
    assert (first.text, first.cached) == ("C E G", False)
    assert (second.text, second.cached) == ("C E G", True)
    client.close()


def test_cache_survives_a_new_client(tmp_path):
    first = CachedLLMClient(FakeLLMClient(replies=["C E G"]), tmp_path)
    first.complete(QUESTION)
    first.close()

    fake = FakeLLMClient(replies=["something else"])
    second = CachedLLMClient(fake, tmp_path)
    assert second.complete(QUESTION).text == "C E G"
    assert fake.calls == []
    second.close()


def test_any_change_in_the_request_is_a_cache_miss(tmp_path):
    fake = FakeLLMClient()
    client = CachedLLMClient(fake, tmp_path)

    client.complete(QUESTION)
    client.complete(QUESTION, temperature=0.7)
    client.complete(QUESTION, system="You are a music teacher.")
    client.complete(QUESTION, max_tokens=10)
    client.complete([Message("user", "Notes of a D minor triad?")])

    assert len(fake.calls) == 5
    client.close()


def test_cache_key_depends_on_model():
    key_a = cache_key("model-a", None, QUESTION, 0.0, 1024)
    key_b = cache_key("model-b", None, QUESTION, 0.0, 1024)
    assert key_a != key_b
    assert key_a == cache_key("model-a", None, QUESTION, 0.0, 1024)


def test_usage_counts_only_fresh_calls():
    usage = UsageTracker()
    usage.record(LLMResponse("a", input_tokens=10, output_tokens=5, latency_s=2.0))
    usage.record(LLMResponse("a", input_tokens=10, output_tokens=5, latency_s=2.0, cached=True))

    assert (usage.calls, usage.cache_hits) == (2, 1)
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)
    assert usage.latency_s == 2.0
    assert "1 cached" in usage.summary()


def test_ollama_client_sends_expected_payload_and_parses_reply():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "C, E, G"},
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        )

    http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    client = OllamaClient("qwen3.5:9b", seed=42, http_client=http)

    reply = client.complete(QUESTION, system="Be brief.", temperature=0.0, max_tokens=50)

    assert seen["path"] == "/api/chat"
    body = seen["body"]
    assert body["model"] == "qwen3.5:9b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["messages"][1] == {"role": "user", "content": "Notes of a C major triad?"}
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_predict": 50}
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("C, E, G", 12, 4)


def test_ollama_client_omits_think_when_disabled():
    client = OllamaClient("some-model", think=None, http_client=httpx.Client())
    payload = client.build_payload(QUESTION, system=None, temperature=0.0, max_tokens=10)
    assert "think" not in payload
    assert payload["messages"] == [{"role": "user", "content": "Notes of a C major triad?"}]
