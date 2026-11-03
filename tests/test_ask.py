from musicteacher.agent.tools import build_registry
from musicteacher.ask import agent_answer, format_sources, rag_answer
from musicteacher.llm.base import ToolCall, ToolTurn
from musicteacher.llm.fake import FakeLLMClient
from musicteacher.retrieval.search import Hit

HIT = Hit(1, 0.9, "en-capo#000", "Capo", "Introduction", "en", "https://w/Capo", "A capo clamps.")


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, mode, k):
        self.calls.append((query, mode, k))
        return [HIT]


def test_rag_answer_retrieves_then_generates_with_the_passages():
    retriever, llm = FakeRetriever(), FakeLLMClient(["A capo raises the pitch [1]."])
    answer, hits = rag_answer("What is a capo?", retriever, llm, "dense", k=5)
    assert answer == "A capo raises the pitch [1]."
    assert retriever.calls == [("What is a capo?", "dense", 5)]
    assert "[1] Capo — Introduction" in llm.calls[0]["messages"][0].content
    assert hits == [HIT]


def test_agent_answer_returns_the_tool_trace_and_searched_sources():
    llm = FakeLLMClient(
        turns=[
            ToolTurn("", [ToolCall("search_corpus", {"query": "capo"})]),
            ToolTurn("A capo clamps the strings [1].", []),
        ]
    )
    answer, result, hits = agent_answer("What is a capo?", llm, build_registry(FakeRetriever()))
    assert answer == "A capo clamps the strings [1]."
    assert [s.tool for s in result.steps] == ["search_corpus"]
    assert hits == [HIT]


def test_format_sources_lists_rank_title_section_and_url():
    assert format_sources([HIT]) == "  [1] Capo — Introduction (en)  https://w/Capo"
