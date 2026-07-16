"""Generalized dataset validators for Fine-Tuning Studio.

Looser than finance-faq-assistant-finetuning's assignment-specific minimums
(50+/100+/50+), since this tool supports arbitrary domains, not one fixed
assignment. Note: unlike that project's tests, the preference validator here
deliberately does NOT require `chosen` to be longer than `rejected` -- that
rule was found during the Finance project's code review to be a length-confound
anti-pattern that risks teaching DPO to prefer verbosity over correctness.
"""

import json

MIN_RAW_TEXT_PARAGRAPHS = 10
MIN_PARAGRAPH_WORDS = 15


def validate_raw_text(content: str) -> list[str]:
    errors = []
    paragraphs = [p.strip() for p in content.strip().split("\n\n") if p.strip()]
    if len(paragraphs) < MIN_RAW_TEXT_PARAGRAPHS:
        errors.append(
            f"Raw text file needs at least {MIN_RAW_TEXT_PARAGRAPHS} paragraphs, "
            f"found {len(paragraphs)}."
        )
    short = [p for p in paragraphs if len(p.split()) < MIN_PARAGRAPH_WORDS]
    if short:
        errors.append(
            f"{len(short)} paragraph(s) have fewer than {MIN_PARAGRAPH_WORDS} words."
        )
    return errors


def _parse_jsonl(content: str) -> tuple[list[dict], list[str]]:
    records = []
    errors = []
    for i, line in enumerate(content.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"Line {i + 1} is not valid JSON: {e}")
    return records, errors


MIN_INSTRUCTION_EXAMPLES = 20
MIN_RESPONSE_WORDS = 5


def validate_instruction_jsonl(content: str) -> list[str]:
    errors = []
    records, parse_errors = _parse_jsonl(content)
    errors.extend(parse_errors)
    if len(records) < MIN_INSTRUCTION_EXAMPLES:
        errors.append(
            f"Instruction dataset needs at least {MIN_INSTRUCTION_EXAMPLES} examples, "
            f"found {len(records)}."
        )
    for i, record in enumerate(records):
        if set(record.keys()) != {"instruction", "response"}:
            errors.append(
                f"Record {i} must have exactly 'instruction' and 'response' keys, "
                f"got {sorted(record.keys())}."
            )
            continue
        if not isinstance(record["instruction"], str) or not record["instruction"].strip():
            errors.append(f"Record {i} has an empty instruction.")
        if (
            not isinstance(record["response"], str)
            or len(record["response"].split()) < MIN_RESPONSE_WORDS
        ):
            errors.append(f"Record {i} has a response with fewer than {MIN_RESPONSE_WORDS} words.")
    return errors
