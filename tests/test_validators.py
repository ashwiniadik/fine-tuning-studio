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


import json

from backend.validators import validate_instruction_jsonl


def _make_instruction_jsonl(n=20, response_words=10):
    lines = []
    for i in range(n):
        response = " ".join(["word"] * response_words)
        lines.append(json.dumps({"instruction": f"Question {i}?", "response": response}))
    return "\n".join(lines)


def test_valid_instruction_jsonl_returns_no_errors():
    assert validate_instruction_jsonl(_make_instruction_jsonl()) == []


def test_instruction_jsonl_too_few_examples():
    errors = validate_instruction_jsonl(_make_instruction_jsonl(n=5))
    assert any("at least 20 examples" in e for e in errors)


def test_instruction_jsonl_bad_schema():
    content = json.dumps({"instruction": "Q?", "answer": "wrong key name"})
    errors = validate_instruction_jsonl(content)
    assert any("exactly 'instruction' and 'response'" in e for e in errors)


def test_instruction_jsonl_short_response():
    content = json.dumps({"instruction": "Q?", "response": "too short"})
    errors = validate_instruction_jsonl(content)
    assert any("fewer than 5 words" in e for e in errors)


def test_instruction_jsonl_invalid_json_line():
    errors = validate_instruction_jsonl("not valid json")
    assert any("not valid JSON" in e for e in errors)
