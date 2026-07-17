"""Shared cell-building helpers for notebook templates."""


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
    that could silently drift apart."""
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
