from musicteacher.ingestion.chunking import WhitespaceTokenizer, chunk_fixed, chunk_sections
from musicteacher.ingestion.clean import Section, split_sections

TOK = WhitespaceTokenizer()
META = {"id": "en-test", "language": "en", "title": "Test"}

RAW = """Intro paragraph about modes.

== History ==
Old text.
=== Medieval ===
Church modes.

== References ==
Some book, 1990.
=== Online ===
A website.

== Modern use ==
Jazz uses Dorian.
"""


def test_split_sections_keeps_hierarchy_and_drops_references():
    sections = split_sections(RAW, "en")
    assert [s.path for s in sections] == [
        "Introduction",
        "History",
        "History > Medieval",
        "Modern use",
    ]
    assert all("book" not in s.text and "website" not in s.text for s in sections)


def test_split_sections_drops_french_portal_footer_and_references():
    raw = "Texte.\n\n== Voir aussi ==\nAutre.\n\nAutre ligne\n Portail de la musique\n"
    sections = split_sections(raw, "fr")
    assert [(s.path, s.text) for s in sections] == [("Introduction", "Texte.")]


def test_chunk_fixed_windows_overlap():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_fixed(META, [Section("Introduction", words)], TOK, size=10, overlap=2)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 10
    assert first[-2:] == second[:2]  # overlap of 2 tokens
    assert chunks[-1].text.split()[-1] == "w24"  # nothing lost at the end
    assert [c.chunk_id for c in chunks][:2] == ["en-test#000", "en-test#001"]


def test_chunk_fixed_records_section_of_chunk_start():
    sections = [Section("A", " ".join(["a"] * 10)), Section("B", " ".join(["b"] * 10))]
    chunks = chunk_fixed(META, sections, TOK, size=10, overlap=0)
    assert [c.section for c in chunks] == ["A", "B"]


def test_chunk_sections_never_crosses_a_section_boundary():
    sections = [Section("A", "one two three"), Section("B", "four five six")]
    chunks = chunk_sections(META, sections, TOK, max_tokens=100, min_tokens=1)
    assert [(c.section, c.text) for c in chunks] == [("A", "one two three"), ("B", "four five six")]


def test_chunk_sections_packs_paragraphs_up_to_max():
    paragraphs = "\n".join(" ".join(["x"] * 4) for _ in range(5))  # 5 paragraphs of 4 tokens
    chunks = chunk_sections(META, [Section("A", paragraphs)], TOK, max_tokens=8, min_tokens=1)
    assert [c.n_tokens for c in chunks] == [8, 8, 4]


def test_chunk_sections_splits_long_paragraph_on_sentences():
    paragraph = "One two three. Four five six. Seven eight nine."
    chunks = chunk_sections(META, [Section("A", paragraph)], TOK, max_tokens=6, min_tokens=1)
    assert [c.text for c in chunks] == ["One two three. Four five six.", "Seven eight nine."]


def test_chunk_sections_merges_tiny_chunks_into_previous():
    sections = [Section("A", " ".join(["a"] * 10)), Section("B", "tiny")]
    chunks = chunk_sections(META, sections, TOK, max_tokens=100, min_tokens=3)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("\ntiny")
