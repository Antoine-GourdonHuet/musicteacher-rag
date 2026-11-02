import pytest

from musicteacher.agent.music import interval, key_info, pitch_class, transpose


def test_pitch_class_handles_accidentals():
    assert [pitch_class(n) for n in ["C", "C#", "Db", "B", "Cb", "E#", "Bbb"]] == [
        0,
        1,
        1,
        11,
        11,
        5,
        9,
    ]


def test_transpose_keeps_chord_quality_and_uses_flats():
    assert transpose(["C", "Am", "F", "G"], 3) == ["Eb", "Cm", "Ab", "Bb"]
    assert transpose(["Dm7", "G7", "Cmaj7"], 5) == ["Gm7", "C7", "Fmaj7"]
    assert transpose(["E", "A", "B7"], -1) == ["Eb", "Ab", "Bb7"]
    assert transpose(["C/E"], 2) == ["D/Gb"]


def test_transpose_rejects_garbage():
    with pytest.raises(ValueError):
        transpose(["H7"], 2)


def test_interval_is_ascending_within_an_octave():
    assert interval("C", "G")["name_en"] == "perfect fifth"
    assert interval("E", "C") == {
        "semitones": 8,
        "name_en": "minor sixth",
        "name_fr": "sixte mineure",
    }
    assert interval("A", "Eb")["name_en"] == "tritone"


def test_key_info_major_keys():
    assert key_info("D major") == {
        "key": "D major",
        "scale": ["D", "E", "F#", "G", "A", "B", "C#"],
        "signature": "2 sharps",
        "relative": "B minor",
    }
    assert key_info("Eb")["signature"] == "3 flats"
    assert key_info("F major")["relative"] == "D minor"
    assert key_info("C major")["signature"] == "no sharps or flats"


def test_key_info_minor_keys():
    info = key_info("C minor")
    assert info["scale"] == ["C", "D", "Eb", "F", "G", "Ab", "Bb"]
    assert info["relative"] == "Eb major"
    assert key_info("Bm")["scale"] == ["B", "C#", "D", "E", "F#", "G", "A"]
