import json

from backend.models_config import MODELS
from backend.notebook_templates.stage1 import build_stage1_notebook


def test_build_stage1_notebook_is_valid_and_complete():
    model = MODELS["qwen2.5-0.5b"]
    nb = build_stage1_notebook(
        domain="legal",
        model_id=model["unsloth_model_id"],
        lora=model["lora"],
        learning_rate=model["stage1"]["learning_rate"],
        epochs=model["stage1"]["epochs"],
    )
    assert nb["nbformat"] == 4
    text = json.dumps(nb).lower()
    for required in [
        "non_instruction_data.txt",
        "fastlanguagemodel",
        "get_peft_model",
        "processing_class",
        "sfttrainer",
        "packing = true",
        "save_pretrained",
        "generate",
        "legal",
    ]:
        assert required in text, f"missing: {required}"


def test_build_stage1_notebook_uses_given_model_id():
    model = MODELS["llama-3.2-1b"]
    nb = build_stage1_notebook(
        domain="healthcare",
        model_id=model["unsloth_model_id"],
        lora=model["lora"],
        learning_rate=2e-4,
        epochs=2,
    )
    assert model["unsloth_model_id"] in json.dumps(nb)


from backend.notebook_templates.stage2 import build_stage2_notebook


def test_build_stage2_notebook_continuing_from_stage1():
    model = MODELS["qwen2.5-0.5b"]
    nb = build_stage2_notebook(
        domain="legal",
        model_id=model["unsloth_model_id"],
        lora=model["lora"],
        learning_rate=model["stage2"]["learning_rate"],
        epochs=model["stage2"]["epochs"],
        previous_adapter="stage1_adapter",
    )
    text = json.dumps(nb).lower()
    for required in [
        "instruction_dataset.jsonl",
        "stage1_adapter",
        "fastlanguagemodel",
        "get_peft_model",
        "alpaca_prompt",
        "processing_class",
        "sfttrainer",
        "packing = false",
        "save_pretrained",
        "generate",
    ]:
        assert required in text, f"missing: {required}"


def test_build_stage2_notebook_without_stage1():
    model = MODELS["qwen2.5-0.5b"]
    nb = build_stage2_notebook(
        domain="legal",
        model_id=model["unsloth_model_id"],
        lora=model["lora"],
        learning_rate=2e-4,
        epochs=3,
        previous_adapter=None,
    )
    text = json.dumps(nb)
    assert model["unsloth_model_id"] in text
    assert "stage1_adapter" not in text
