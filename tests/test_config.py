from musicteacher.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.top_k == 5
    assert s.temperature == 0.0
    assert s.seed == 42
