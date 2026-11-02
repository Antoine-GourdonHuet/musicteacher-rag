import json

from musicteacher.agent.loop import NO_ANSWER, run_agent
from musicteacher.agent.tools import build_registry
from musicteacher.evaluation.run_agent import args_match
from musicteacher.llm.base import ToolCall, ToolTurn
from musicteacher.llm.fake import FakeLLMClient
from musicteacher.llm.ollama_client import parse_arguments
from musicteacher.retrieval.search import Hit

REGISTRY = build_registry()  # music tools only


def call(name: str, **arguments) -> ToolTurn:
    return ToolTurn(content="", tool_calls=[ToolCall(name, arguments)])


def text(content: str) -> ToolTurn:
    return ToolTurn(content=content, tool_calls=[])


def test_registry_runs_tools_and_turns_errors_into_messages():
    assert (
        REGISTRY.execute("transpose", {"chords": ["C", "Am"], "semitones": 3}).content == "Eb - Cm"
    )
    assert "septième majeure" in REGISTRY.execute("interval", {"note1": "F", "note2": "E"}).content
    assert "3 flats" in REGISTRY.execute("key_info", {"key": "Eb major"}).content

    unknown = REGISTRY.execute("teleport", {})
    assert not unknown.ok and "unknown tool" in unknown.content
    invalid = REGISTRY.execute("transpose", {"chords": [], "semitones": 30})
    assert not invalid.ok and "invalid arguments" in invalid.content
    failing = REGISTRY.execute("transpose", {"chords": ["H7"], "semitones": 2})
    assert not failing.ok and "transpose failed" in failing.content


def test_agent_calls_a_tool_then_answers_with_its_result():
    llm = FakeLLMClient(
        turns=[call("transpose", chords=["C", "G"], semitones=2), text("D - A, a tone higher.")]
    )
    result = run_agent(llm, REGISTRY, "Transpose C - G up a tone.")

    assert result.answer == "D - A, a tone higher."
    assert [(s.tool, s.ok, s.result) for s in result.steps] == [("transpose", True, "D - A")]
    assert result.turns == 2 and result.stopped == "answer"
    second_turn = llm.calls[1]["messages"]
    assert second_turn[-2]["tool_calls"][0]["function"]["name"] == "transpose"
    assert second_turn[-1] == {"role": "tool", "tool_name": "transpose", "content": "D - A"}


def test_tool_error_is_sent_back_and_the_model_can_recover():
    llm = FakeLLMClient(
        turns=[
            call("transpose", chords=["C"], semitones=40),  # invalid
            call("transpose", chords=["C"], semitones=4),
            text("E"),
        ]
    )
    result = run_agent(llm, REGISTRY, "Transpose C up a major third.")
    assert [s.ok for s in result.steps] == [False, True]
    assert result.answer == "E"


def test_max_steps_falls_back_to_the_last_successful_tool_result():
    looping = [call("interval", note1="C", note2="G")] * 5
    result = run_agent(FakeLLMClient(turns=looping), REGISTRY, "C to G?", max_steps=3)
    assert result.stopped == "max_steps"
    assert result.turns == 3
    assert result.answer.startswith("perfect fifth")


def test_empty_answer_without_tools_uses_the_fixed_fallback():
    result = run_agent(FakeLLMClient(turns=[text("   ")]), REGISTRY, "?")
    assert (result.answer, result.stopped) == (NO_ANSWER, "empty_answer")


def test_search_tool_returns_numbered_passages():
    class FakeRetriever:
        def search(self, query, mode, k):
            return [Hit(1, 1.0, "en-capo#000", "Capo", "Introduction", "en", "u", "A capo clamps.")]

    registry = build_registry(FakeRetriever(), k=5)
    result = registry.execute("search_corpus", {"query": "what is a capo"})
    assert result.content == "[1] Capo — Introduction\nA capo clamps."
    assert result.data[0].chunk_id == "en-capo#000"
    assert list(registry.tools)[0] == "search_corpus"


def test_args_match_is_tolerant_to_equivalent_forms():
    expected = {"chords": ["C", "Am"], "semitones": -2}
    assert args_match("transpose", expected, {"chords": ["C", "Am"], "semitones": 10})
    assert not args_match("transpose", expected, {"chords": ["C", "Am"], "semitones": 2})
    assert args_match("interval", {"note1": "A", "note2": "Eb"}, {"note1": "A", "note2": "D#"})
    assert args_match("key_info", {"key": "C minor"}, {"key": "Cm"})
    assert not args_match("key_info", {"key": "C minor"}, {"key": "C major"})
    assert not args_match("transpose", expected, {"chords": "oops"})


def test_tool_arguments_can_arrive_as_a_json_string():
    assert parse_arguments(json.dumps({"key": "D major"})) == {"key": "D major"}
    assert parse_arguments("not json") == {}
    assert parse_arguments(None) == {}
