from backend.validators import validate_raw_text


def _make_raw_text(n_paragraphs=10, words_per_paragraph=15):
    paragraph = " ".join(["word"] * words_per_paragraph)
    return "\n\n".join([paragraph] * n_paragraphs)


def test_valid_raw_text_returns_no_errors():
    assert validate_raw_text(_make_raw_text()) == []


def test_raw_text_too_few_paragraphs():
    errors = validate_raw_text(_make_raw_text(n_paragraphs=5))
    assert any("at least 10 paragraphs" in e for e in errors)


def test_raw_text_paragraph_too_short():
    text = _make_raw_text(n_paragraphs=9) + "\n\nToo short."
    errors = validate_raw_text(text)
    assert any("fewer than 15 words" in e for e in errors)
