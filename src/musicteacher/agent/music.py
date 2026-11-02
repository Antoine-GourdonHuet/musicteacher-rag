"""Deterministic music-theory functions used as agent tools (and to compute reference answers).

Conventions:
- Transposed chord roots are spelled with flats (Eb, not D#), as in the review guide.
- Scales are spelled correctly (one letter per degree): D major contains F# and C#, not Gb and Db.
- Intervals are named from their size in semitones, ascending from the first note to the second.
"""

import re

LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTERS = "CDEFGAB"
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
ACCIDENTALS = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
NOTE = re.compile(r"^([A-G])(#{1,2}|b{1,2})?$")
CHORD = re.compile(r"^([A-G](?:#|b)?)([^/]*)(?:/([A-G](?:#|b)?))?$")

INTERVALS = [
    ("unison", "unisson"),
    ("minor second", "seconde mineure"),
    ("major second", "seconde majeure"),
    ("minor third", "tierce mineure"),
    ("major third", "tierce majeure"),
    ("perfect fourth", "quarte juste"),
    ("tritone", "triton"),
    ("perfect fifth", "quinte juste"),
    ("minor sixth", "sixte mineure"),
    ("major sixth", "sixte majeure"),
    ("minor seventh", "septième mineure"),
    ("major seventh", "septième majeure"),
]
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]


def pitch_class(note: str) -> int:
    match = NOTE.match(note.strip())
    if not match:
        raise ValueError(f"not a note name: {note!r} (expected e.g. C, F#, Bb)")
    letter, accidental = match.group(1), match.group(2) or ""
    return (LETTER_PC[letter] + accidental.count("#") - accidental.count("b")) % 12


def transpose_chord(chord: str, semitones: int) -> str:
    """Transpose one chord symbol, keeping its quality: Am7 +3 -> Cm7, C/E +2 -> D/Gb."""
    match = CHORD.match(chord.strip())
    if not match:
        raise ValueError(f"not a chord symbol: {chord!r}")
    root, quality, bass = match.groups()
    result = FLAT_NAMES[(pitch_class(root) + semitones) % 12] + quality
    if bass:
        result += "/" + FLAT_NAMES[(pitch_class(bass) + semitones) % 12]
    return result


def transpose(chords: list[str], semitones: int) -> list[str]:
    return [transpose_chord(c, semitones) for c in chords]


def interval(note1: str, note2: str) -> dict:
    """Ascending interval from note1 to note2 (within one octave)."""
    semitones = (pitch_class(note2) - pitch_class(note1)) % 12
    name_en, name_fr = INTERVALS[semitones]
    return {"semitones": semitones, "name_en": name_en, "name_fr": name_fr}


def spell_scale(tonic: str, steps: list[int]) -> list[str]:
    """One note per letter from the tonic's letter, each with the accidental its degree needs."""
    start = LETTERS.index(tonic[0])
    tonic_pc = pitch_class(tonic)
    notes = []
    for degree, step in enumerate(steps):
        letter = LETTERS[(start + degree) % 7]
        offset = (tonic_pc + step - LETTER_PC[letter]) % 12
        offset = offset - 12 if offset > 6 else offset
        if offset not in ACCIDENTALS:
            raise ValueError(f"{tonic} needs a triple accidental: use an enharmonic key")
        notes.append(letter + ACCIDENTALS[offset])
    return notes


def parse_key(key: str) -> tuple[str, str]:
    """'Eb major', 'C minor', 'Cm', 'F#' -> (tonic, 'major' | 'minor')."""
    text = key.strip()
    match = re.match(r"^([A-G](?:#|b)?)\s*(major|minor|maj|min|m)?$", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"not a key: {key!r} (expected e.g. 'Eb major' or 'C minor')")
    tonic, mode = match.group(1), (match.group(2) or "major").lower()
    return tonic, "minor" if mode in ("minor", "min", "m") else "major"


def key_info(key: str) -> dict:
    """Scale notes, key signature and relative key of a major or natural minor key."""
    tonic, mode = parse_key(key)
    scale = spell_scale(tonic, MAJOR_STEPS if mode == "major" else NATURAL_MINOR_STEPS)
    sharps = sum(n.count("#") for n in scale)
    flats = sum(n.count("b") for n in scale)
    if sharps and flats:
        raise ValueError(f"{key} is not a standard key: use its enharmonic equivalent")
    if mode == "major":
        relative = f"{scale[5]} minor"
    else:
        relative = f"{scale[2]} major"
    if sharps:
        signature = f"{sharps} sharp" + ("s" if sharps > 1 else "")
    elif flats:
        signature = f"{flats} flat" + ("s" if flats > 1 else "")
    else:
        signature = "no sharps or flats"
    return {"key": f"{tonic} {mode}", "scale": scale, "signature": signature, "relative": relative}
