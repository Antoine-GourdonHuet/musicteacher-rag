"""The agent loop: the model decides which tool to call, we run it, and feed the result back.

    question -> model turn -> tool call(s)? -> run tools -> results back to the model -> ...
                           -> plain text?   -> that is the answer

Safeguards:
- `max_steps` model turns at most (a model can loop on tool calls forever);
- tools never raise (see `tools.ToolRegistry.execute`): errors are sent back as text;
- deterministic fallback if the model never gives a text answer: return the last successful tool
  result as is (for computation questions it is the answer), else a fixed "cannot answer" message.
"""

from dataclasses import dataclass, field
from typing import Any

from musicteacher.agent.tools import ToolRegistry
from musicteacher.llm.base import ToolLLMClient

AGENT_SYSTEM = """You are a music theory and guitar teaching assistant with tools.
- For any computation (transposing chords, naming an interval, the notes, key signature or
  relative key of a key), call the matching tool: transpose, interval or key_info.
  Never compute it yourself.
- For any other music theory, guitar or singing question, call search_corpus, then answer ONLY
  from the passages it returns and cite them by number, for example [2]. If they do not contain
  the answer, say that you do not know.
- Answer in the same language as the question, in 1 to 4 sentences.
- Write notes with flats as the tools do (Eb, not D#)."""

NO_ANSWER = "I could not answer this question with the available tools."


@dataclass
class AgentStep:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: str
    data: Any = None  # not serialised: structured tool output (e.g. retrieved passages)


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    turns: int = 0
    stopped: str = "answer"  # "answer" | "max_steps" | "empty_answer"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


def fallback_answer(steps: list[AgentStep]) -> str:
    successful = [s for s in steps if s.ok]
    return successful[-1].result if successful else NO_ANSWER


def run_agent(
    llm: ToolLLMClient,
    registry: ToolRegistry,
    question: str,
    *,
    max_steps: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> AgentResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    result = AgentResult(answer="")
    for _ in range(max_steps):
        turn = llm.chat_with_tools(
            messages,
            registry.schemas(),
            system=AGENT_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result.turns += 1
        result.input_tokens += turn.input_tokens
        result.output_tokens += turn.output_tokens
        result.latency_s += turn.latency_s

        if not turn.tool_calls:  # the model answers in text: done
            if turn.content.strip():
                result.answer = turn.content.strip()
            else:
                result.answer, result.stopped = fallback_answer(result.steps), "empty_answer"
            return result

        messages.append(
            {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in turn.tool_calls
                ],
            }
        )
        for call in turn.tool_calls:
            outcome = registry.execute(call.name, call.arguments)
            result.steps.append(
                AgentStep(call.name, call.arguments, outcome.ok, outcome.content, outcome.data)
            )
            messages.append({"role": "tool", "tool_name": call.name, "content": outcome.content})

    result.answer, result.stopped = fallback_answer(result.steps), "max_steps"
    return result
