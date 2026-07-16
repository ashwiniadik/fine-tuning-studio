"""Generalized dataset validators for Fine-Tuning Studio.

Looser than finance-faq-assistant-finetuning's assignment-specific minimums
(50+/100+/50+), since this tool supports arbitrary domains, not one fixed
assignment. Note: unlike that project's tests, the preference validator here
deliberately does NOT require `chosen` to be longer than `rejected` -- that
rule was found during the Finance project's code review to be a length-confound
anti-pattern that risks teaching DPO to prefer verbosity over correctness.
"""

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
