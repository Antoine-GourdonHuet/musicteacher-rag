"""Clean raw Wikipedia plain text and split it into sections.

The raw extracts keep headings as `== Heading ==` (level 2) to `====== Heading ======` (level 6).
We keep the heading hierarchy as a section path ("History > Modern use"), drop reference-like
sections (references, see also, external links...) and the French "Portail" footer.
"""

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
PORTAL_FOOTER = re.compile(r"^Portail (de|du|des) ")
INTRO = "Introduction"

DROPPED_SECTIONS = {
    "en": {
        "references",
        "notes",
        "notes and references",
        "citations",
        "footnotes",
        "sources",
        "bibliography",
        "further reading",
        "see also",
        "external links",
    },
    "fr": {
        "notes et références",
        "références",
        "notes",
        "sources",
        "bibliographie",
        "voir aussi",
        "articles connexes",
        "liens externes",
        "annexes",
    },
}


@dataclass(frozen=True)
class Section:
    path: str  # e.g. "Dorian mode > Modern Dorian mode", or "Introduction"
    text: str  # paragraphs separated by "\n"


def split_sections(text: str, lang: str) -> list[Section]:
    dropped = DROPPED_SECTIONS[lang]
    sections: list[Section] = []
    headings: list[str] = []
    paragraphs: list[str] = []
    path = INTRO
    skip_level: int | None = None  # set while inside a dropped section

    def flush() -> None:
        if paragraphs:
            sections.append(Section(path, "\n".join(paragraphs)))
        paragraphs.clear()

    for line in text.splitlines():
        line = line.strip()
        match = HEADING.match(line)
        if match:
            level, title = len(match.group(1)), match.group(2)
            if skip_level is not None and level > skip_level:
                continue  # sub-heading of a dropped section
            flush()
            headings = headings[: level - 2] + [title]
            path = " > ".join(headings)
            skip_level = level if title.lower() in dropped else None
            continue
        if skip_level is None and line and not PORTAL_FOOTER.match(line):
            paragraphs.append(line)
    flush()
    return sections
