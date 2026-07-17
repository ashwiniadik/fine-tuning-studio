"""Ties together validation, stage selection, and notebook generation."""

import json

from backend.models_config import MODELS
from backend.notebook_templates.stage1 import build_stage1_notebook
from backend.notebook_templates.stage2 import build_stage2_notebook
from backend.notebook_templates.stage3 import build_stage3_notebook
from backend.validators import validate_instruction_jsonl, validate_preference_jsonl, validate_raw_text


def select_stages(has_raw_text: bool, has_preference: bool) -> list[str]:
    """Instruction data (Stage 2) is always required and always included.
    Raw text (Stage 1) and preference data (Stage 3) are optional add-ons."""
    stages = []
    if has_raw_text:
        stages.append("stage1")
    stages.append("stage2")
    if has_preference:
        stages.append("stage3")
    return stages


class ValidationFailed(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def build_readme(domain: str, model_display_name: str, stages: list[str]) -> str:
    return f"# {domain.title()} Fine-Tuning Project\n"


def generate_project(
    domain: str,
    model_key: str,
    instruction_data: str,
    raw_text: str | None = None,
    preference_data: str | None = None,
) -> dict[str, bytes]:
    errors = [f"instruction_dataset.jsonl: {e}" for e in validate_instruction_jsonl(instruction_data)]
    if raw_text is not None:
        errors += [f"non_instruction_data.txt: {e}" for e in validate_raw_text(raw_text)]
    if preference_data is not None:
        errors += [f"preference_dataset.jsonl: {e}" for e in validate_preference_jsonl(preference_data)]
    if model_key not in MODELS:
        errors.append(f"Unknown model '{model_key}'. Choose one of: {', '.join(MODELS)}.")
    if errors:
        raise ValidationFailed(errors)

    model = MODELS[model_key]
    stages = select_stages(has_raw_text=raw_text is not None, has_preference=preference_data is not None)

    files: dict[str, bytes] = {
        "data/instruction_dataset.jsonl": instruction_data.encode("utf-8"),
    }
    if raw_text is not None:
        files["data/non_instruction_data.txt"] = raw_text.encode("utf-8")
    if preference_data is not None:
        files["data/preference_dataset.jsonl"] = preference_data.encode("utf-8")

    previous_adapter = None
    if "stage1" in stages:
        nb1 = build_stage1_notebook(
            domain, model["unsloth_model_id"], model["lora"],
            model["stage1"]["learning_rate"], model["stage1"]["epochs"],
        )
        files["notebooks/non_instruction_finetuning.ipynb"] = json.dumps(nb1, indent=1).encode("utf-8")
        previous_adapter = "stage1_adapter"

    nb2 = build_stage2_notebook(
        domain, model["unsloth_model_id"], model["lora"],
        model["stage2"]["learning_rate"], model["stage2"]["epochs"], previous_adapter,
    )
    files["notebooks/instruction_finetuning.ipynb"] = json.dumps(nb2, indent=1).encode("utf-8")

    if "stage3" in stages:
        nb3 = build_stage3_notebook(
            domain, model["unsloth_model_id"], model["lora"],
            model["stage3"]["learning_rate"], model["stage3"]["epochs"], "stage2_adapter",
        )
        files["notebooks/dpo_alignment.ipynb"] = json.dumps(nb3, indent=1).encode("utf-8")

    files["README.md"] = build_readme(domain, model["display_name"], stages).encode("utf-8")
    return files
