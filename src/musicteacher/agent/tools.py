"""Tools the agent can call: a JSON schema for the model + a validated Python implementation.

Robustness rules: a tool never raises into the agent loop. Unknown tool, invalid arguments
(checked with pydantic) or an exception inside the tool all become an "ERROR: ..." result that is
sent back to the model, which can then retry or answer.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from musicteacher.agent.music import interval, key_info, transpose
from musicteacher.generation.rag import format_context
from musicteacher.retrieval.search import Retriever


class SearchArgs(BaseModel):
    query: str = Field(min_length=3)


class TransposeArgs(BaseModel):
    chords: list[str] = Field(min_length=1)
    semitones: int = Field(ge=-11, le=11)


class IntervalArgs(BaseModel):
    note1: str
    note2: str


class KeyInfoArgs(BaseModel):
    key: str


@dataclass
class ToolResult:
    ok: bool
    content: str  # what the model sees
    data: Any = None  # structured result kept for evaluation (e.g. retrieved passages)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema shown to the model
    args_model: type[BaseModel]
    run: Callable[[BaseModel], ToolResult]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            available = ", ".join(self.tools)
            return ToolResult(False, f"ERROR: unknown tool '{name}'. Available tools: {available}.")
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult(False, f"ERROR: invalid arguments for {name}: {problems}")
        try:
            return tool.run(args)
        except Exception as exc:  # a tool bug must not crash the agent loop
            return ToolResult(False, f"ERROR: {name} failed: {exc}")


def _transpose(args: TransposeArgs) -> ToolResult:
    result = transpose(args.chords, args.semitones)
    return ToolResult(True, " - ".join(result), result)


def _interval(args: IntervalArgs) -> ToolResult:
    result = interval(args.note1, args.note2)
    content = (
        f"{result['name_en']} ({result['semitones']} semitones); "
        f"en français : {result['name_fr']} ({result['semitones']} demi-tons)"
    )
    return ToolResult(True, content, result)


def _key_info(args: KeyInfoArgs) -> ToolResult:
    info = key_info(args.key)
    content = (
        f"{info['key']}: scale {' '.join(info['scale'])}; key signature: {info['signature']}; "
        f"relative key: {info['relative']}"
    )
    return ToolResult(True, content, info)


def _search(retriever: Retriever, mode: str, k: int) -> Callable[[SearchArgs], ToolResult]:
    def run(args: SearchArgs) -> ToolResult:
        hits = retriever.search(args.query, mode, k=k)
        return ToolResult(True, format_context(hits), hits)

    return run


NOTE_HELP = "Letter name with optional # or b, e.g. C, F#, Bb."

MUSIC_TOOLS = [
    Tool(
        name="transpose",
        description=(
            "Transpose chord symbols by a number of semitones, keeping each chord's quality. "
            "Positive semitones = up, negative = down. Examples: up a minor third = 3, "
            "down a whole tone = -2, up a perfect fifth = 7, down a semitone = -1."
        ),
        parameters={
            "type": "object",
            "properties": {
                "chords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chord symbols in order, e.g. ['C', 'Am', 'F', 'G7'].",
                },
                "semitones": {
                    "type": "integer",
                    "description": "Between -11 and 11. Positive = up, negative = down.",
                },
            },
            "required": ["chords", "semitones"],
        },
        args_model=TransposeArgs,
        run=_transpose,
    ),
    Tool(
        name="interval",
        description="Name the ascending interval from note1 up to note2, within one octave.",
        parameters={
            "type": "object",
            "properties": {
                "note1": {"type": "string", "description": f"Lower note. {NOTE_HELP}"},
                "note2": {"type": "string", "description": f"Upper note. {NOTE_HELP}"},
            },
            "required": ["note1", "note2"],
        },
        args_model=IntervalArgs,
        run=_interval,
    ),
    Tool(
        name="key_info",
        description=(
            "Scale notes, key signature (number of sharps or flats) and relative key of a "
            "major or natural minor key."
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Tonic and mode, e.g. 'Eb major', 'C minor', 'F# minor'.",
                }
            },
            "required": ["key"],
        },
        args_model=KeyInfoArgs,
        run=_key_info,
    ),
]


def build_registry(
    retriever: Retriever | None = None, mode: str = "hybrid_rerank", k: int = 5
) -> ToolRegistry:
    """Music tools, plus `search_corpus` when a retriever is given."""
    tools = list(MUSIC_TOOLS)
    if retriever is not None:
        tools.insert(
            0,
            Tool(
                name="search_corpus",
                description=(
                    "Search the music theory, guitar and singing knowledge base (Wikipedia "
                    "articles in English and French). Returns numbered passages to cite."
                ),
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "What to look up."}},
                    "required": ["query"],
                },
                args_model=SearchArgs,
                run=_search(retriever, mode, k),
            ),
        )
    return ToolRegistry({t.name: t for t in tools})
