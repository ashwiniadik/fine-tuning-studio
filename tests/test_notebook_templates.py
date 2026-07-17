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


from backend.notebook_templates.stage3 import build_stage3_notebook


def test_build_stage3_notebook_is_valid_and_complete():
    model = MODELS["qwen2.5-0.5b"]
    nb = build_stage3_notebook(
        domain="legal",
        model_id=model["unsloth_model_id"],
        lora=model["lora"],
        learning_rate=model["stage3"]["learning_rate"],
        epochs=model["stage3"]["epochs"],
        previous_adapter="stage2_adapter",
    )
    text = json.dumps(nb).lower()
    for required in [
        "preference_dataset.jsonl",
        "stage2_adapter",
        "dpotrainer",
        "dpoconfig",
        "ref_model",
        "save_pretrained",
        "generate",
    ]:
        assert required in text, f"missing: {required}"


def test_stage2_and_stage3_use_identical_alpaca_prompt():
    # Guards the exact bug class found in finance-faq-assistant-finetuning: DPO
    # training must compare responses under the same prompt format the model
    # was actually SFT-trained on, or the preference signal doesn't transfer.
    from backend.notebook_templates.stage2 import build_stage2_notebook

    model = MODELS["qwen2.5-0.5b"]
    nb2 = build_stage2_notebook(
        "legal", model["unsloth_model_id"], model["lora"], 2e-4, 3, "stage1_adapter"
    )
    nb3 = build_stage3_notebook(
        "legal", model["unsloth_model_id"], model["lora"], 5e-6, 2, "stage2_adapter"
    )
    from backend.notebook_templates._helpers import alpaca_prompt_text

    prompt_text = alpaca_prompt_text("legal")
    # json.dumps escapes real newlines as literal "\n", so compare against the
    # same escaped form rather than the raw (real-newline) prompt text.
    escaped_prompt_text = prompt_text.replace("\n", "\\n")
    assert escaped_prompt_text in json.dumps(nb2)
    assert escaped_prompt_text in json.dumps(nb3)


def test_alpaca_prompt_text_rejects_syntax_breaking_domain_directly():
    # Defense in depth: alpaca_prompt_text is a shared helper that any future
    # caller could invoke directly, bypassing the FastAPI-level validation in
    # backend/app.py. It must refuse to build a broken prompt template rather
    # than silently emitting Python source that fails to compile or crashes
    # at .format() runtime.
    import pytest

    from backend.notebook_templates._helpers import alpaca_prompt_text

    with pytest.raises(ValueError):
        alpaca_prompt_text('legal"""')
    with pytest.raises(ValueError):
        alpaca_prompt_text("legal {} braces")


def test_build_stage2_notebook_rejects_syntax_breaking_domain():
    # End-to-end: building a Stage 2 notebook with a domain containing `"""`
    # must raise a clean error rather than emitting a code cell that fails to
    # compile() in Colab (the originally reported bug).
    import pytest

    from backend.notebook_templates.stage2 import build_stage2_notebook

    model = MODELS["qwen2.5-0.5b"]
    with pytest.raises(ValueError):
        build_stage2_notebook(
            'legal"""', model["unsloth_model_id"], model["lora"], 2e-4, 3, None
        )
