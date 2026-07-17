import json

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


def test_instruction_jsonl_non_object_line_reports_clean_error():
    # A line that's valid JSON but not an object (e.g. a bare list) must not
    # crash with an AttributeError on record.keys() -- it should produce a
    # clean validation error like every other malformed-input case.
    errors = validate_instruction_jsonl("[1, 2, 3]")
    assert any("must be a JSON object" in e for e in errors)


from backend.validators import validate_preference_jsonl


def _make_preference_jsonl(n=20):
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "prompt": f"Question {i}?",
                    "chosen": "A specific, correct, helpful answer.",
                    "rejected": "A vague non-answer.",
                }
            )
        )
    return "\n".join(lines)


def test_valid_preference_jsonl_returns_no_errors():
    assert validate_preference_jsonl(_make_preference_jsonl()) == []


def test_preference_jsonl_too_few_examples():
    errors = validate_preference_jsonl(_make_preference_jsonl(n=5))
    assert any("at least 20 examples" in e for e in errors)


def test_preference_jsonl_bad_schema():
    content = json.dumps({"prompt": "Q?", "chosen": "A", "rejected": "B", "extra": "C"})
    errors = validate_preference_jsonl(content)
    assert any("exactly 'prompt', 'chosen', and 'rejected'" in e for e in errors)


def test_preference_jsonl_non_string_chosen_or_rejected_reports_clean_error():
    # A record where `chosen` or `rejected` is not a string (e.g. a number)
    # must not crash with an AttributeError on .strip() -- it should produce
    # a clean validation error like every other malformed-input case.
    content = json.dumps({"prompt": "Q?", "chosen": 5, "rejected": "b"})
    errors = validate_preference_jsonl(content)
    assert any("non-string chosen or rejected" in e for e in errors)


def test_preference_jsonl_identical_chosen_and_rejected():
    content = json.dumps({"prompt": "Q?", "chosen": "Same text.", "rejected": "Same text."})
    errors = validate_preference_jsonl(content)
    assert any("identical chosen and rejected" in e for e in errors)


def test_preference_jsonl_does_not_require_chosen_longer_than_rejected():
    # Deliberate design decision: the Finance project's tests required chosen
    # to be a longer word count than rejected, but that rule was flagged during
    # that project's code review as a length-confound anti-pattern for DPO
    # training. This validator intentionally does not enforce it -- a short,
    # sharp chosen answer must not be rejected by the validator just for being
    # shorter than a padded-out rejected one.
    content = json.dumps(
        {
            "prompt": "What is a SIP?",
            "chosen": "A SIP is a plan.",
            "rejected": (
                "A systematic investment plan lets you invest a fixed amount "
                "regularly into a mutual fund over an extended period of time."
            ),
        }
    )
    errors = validate_preference_jsonl(content)
    assert not any("longer" in e.lower() for e in errors)


from backend.validators import validate_domain


def test_valid_simple_domain_returns_no_errors():
    assert validate_domain("legal") == []


def test_valid_multiword_hyphenated_domains_return_no_errors():
    # Real, reasonable domain names must not be rejected -- the threat model
    # is "prevent syntax-breaking characters," not "only allow single words."
    for domain in ("customer support", "e-commerce", "K-12 education", "Tax & Accounting"):
        assert validate_domain(domain) == [], f"domain {domain!r} unexpectedly rejected"


def test_domain_empty_is_rejected():
    errors = validate_domain("   ")
    assert any("must not be empty" in e for e in errors)


def test_domain_with_triple_quote_is_rejected():
    # `domain` gets embedded into a Python triple-quoted string literal in
    # generated notebooks; `"""` would produce a notebook cell that fails to
    # even compile (SyntaxError: unterminated triple-quoted string literal).
    errors = validate_domain('legal"""')
    assert len(errors) > 0


def test_domain_with_braces_is_rejected():
    # `{` / `}` compile fine but break the generated notebook's
    # ALPACA_PROMPT.format() call at runtime (IndexError: Replacement index
    # out of range for positional args tuple).
    errors = validate_domain("legal {} braces")
    assert len(errors) > 0


def test_domain_too_long_is_rejected():
    errors = validate_domain("a" * 101)
    assert any("100 characters" in e for e in errors)
