"""Shared cell-building helpers for notebook templates."""

from backend.validators import validate_domain


def markdown(source: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: list[str]) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def lora_target_modules_literal(target_modules: list[str]) -> str:
    return ", ".join(f'"{m}"' for m in target_modules)


def alpaca_prompt_text(domain: str) -> str:
    """The exact Alpaca-style prompt template text, shared by every template
    that needs it (Stage 2 SFT and Stage 3 DPO), so both are guaranteed to use
    an identical string for a given domain rather than two hand-written copies
    that could silently drift apart.

    `domain` ends up embedded inside a triple-quoted Python string literal
    (ALPACA_PROMPT = triple-quote + prompt_text + triple-quote) in the
    generated notebook, which is later fed through `.format()`. It's expected
    to already be validated at the API boundary (see validators.validate_domain),
    but this is checked again here so any caller of this function -- not just
    the FastAPI route -- gets a clean error instead of a notebook that fails
    to parse or crashes at .format() time."""
    errors = validate_domain(domain)
    if errors:
        raise ValueError("; ".join(errors))
    return (
        f"Below is an instruction that describes a {domain} question. "
        "Write a response that answers it accurately and clearly.\n"
        "\n"
        "### Instruction:\n"
        "{}\n"
        "\n"
        "### Response:\n"
        "{}"
    )
