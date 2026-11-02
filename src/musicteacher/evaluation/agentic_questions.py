"""The 20 agentic questions: written from templates, reference answers computed by the music tools.

Not drafted by the LLM: a wrong reference answer on a computation question would silently corrupt
the measurement. Each answer must still be checked by hand during review (independent check: the
tool and the reference come from the same code).
"""

from musicteacher.agent.music import interval, key_info, transpose
from musicteacher.evaluation.dataset import EvalQuestion

# (lang, question, chords, semitones)
TRANSPOSITIONS = [
    ("en", "Transpose the progression C - Am - F - G up a minor third.", ["C", "Am", "F", "G"], 3),
    ("en", "Transpose G - Em - C - D down a whole tone.", ["G", "Em", "C", "D"], -2),
    ("en", "Transpose Dm7 - G7 - Cmaj7 up a perfect fourth.", ["Dm7", "G7", "Cmaj7"], 5),
    ("en", "Transpose Bb - Eb - F up two semitones.", ["Bb", "Eb", "F"], 2),
    (
        "fr",
        "Transpose la progression Ab - Eb - Fm - Db d'une tierce majeure vers le haut.",
        ["Ab", "Eb", "Fm", "Db"],
        4,
    ),
    ("fr", "Transpose la grille E - A - B7 un demi-ton plus bas.", ["E", "A", "B7"], -1),
    ("fr", "Transpose Am - Dm - E7 d'une quinte juste vers le haut.", ["Am", "Dm", "E7"], 7),
    (
        "fr",
        "Transpose la grille F - Bb - C7 de trois demi-tons vers le bas.",
        ["F", "Bb", "C7"],
        -3,
    ),
]

# (lang, question, note1, note2)
INTERVALS = [
    ("en", "What interval is there from C up to G?", "C", "G"),
    ("en", "Name the ascending interval from E to C.", "E", "C"),
    ("en", "What interval separates A and the Eb above it?", "A", "Eb"),
    ("fr", "Quel intervalle y a-t-il de D à F en montant ?", "D", "F"),
    ("fr", "Quel est l'intervalle ascendant de F à E ?", "F", "E"),
    ("fr", "Quel intervalle sépare Bb du D situé au-dessus ?", "Bb", "D"),
]

# (lang, question, key, field of key_info that answers it)
KEYS = [
    ("en", "How many flats are in the key signature of Eb major?", "Eb major", "signature"),
    ("en", "What is the relative minor of F major?", "F major", "relative"),
    ("en", "Which notes make up the D major scale?", "D major", "scale"),
    ("fr", "Combien d'altérations a l'armure de A major (la majeur) ?", "A major", "signature"),
    ("fr", "Quel est le relatif majeur de C minor (do mineur) ?", "C minor", "relative"),
    ("fr", "Quelles notes composent la gamme de B minor naturelle ?", "B minor", "scale"),
]


FRENCH_TERMS = [
    ("no sharps or flats", "aucune altération"),
    ("sharps", "dièses"),
    ("sharp", "dièse"),
    ("flats", "bémols"),
    ("flat", "bémol"),
    ("major", "majeur"),
    ("minor", "mineur"),
]


def to_french(answer: str) -> str:
    for en, fr in FRENCH_TERMS:
        answer = answer.replace(en, fr)
    return answer


def agentic_questions() -> list[EvalQuestion]:
    questions = []

    def add(lang: str, question: str, answer: str, tool: str, args: dict) -> None:
        questions.append(
            EvalQuestion(
                id=f"ag-{len(questions) + 1:03d}",
                lang=lang,
                qtype="agentic",
                question=question,
                reference_answer=answer,
                origin="template",
                expected_tool=tool,
                expected_tool_args=args,
            )
        )

    for lang, question, chords, semitones in TRANSPOSITIONS:
        answer = " - ".join(transpose(chords, semitones))
        add(lang, question, answer, "transpose", {"chords": chords, "semitones": semitones})
    for lang, question, note1, note2 in INTERVALS:
        result = interval(note1, note2)
        if lang == "fr":
            answer = f"{result['name_fr']} ({result['semitones']} demi-tons)"
        else:
            answer = f"{result['name_en']} ({result['semitones']} semitones)"
        add(lang, question, answer, "interval", {"note1": note1, "note2": note2})
    for lang, question, key, field in KEYS:
        value = key_info(key)[field]
        answer = " ".join(value) if isinstance(value, list) else value
        if lang == "fr":
            answer = to_french(answer)
        add(lang, question, answer, "key_info", {"key": key})
    return questions
