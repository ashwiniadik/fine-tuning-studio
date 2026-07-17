from backend.generator import select_stages


def test_instruction_only():
    assert select_stages(has_raw_text=False, has_preference=False) == ["stage2"]


def test_raw_text_and_instruction():
    assert select_stages(has_raw_text=True, has_preference=False) == ["stage1", "stage2"]


def test_instruction_and_preference():
    assert select_stages(has_raw_text=False, has_preference=True) == ["stage2", "stage3"]


def test_all_three():
    assert select_stages(has_raw_text=True, has_preference=True) == ["stage1", "stage2", "stage3"]


import json

from backend.generator import ValidationFailed, generate_project


def _instruction_data(n=20):
    return "\n".join(
        json.dumps({"instruction": f"Q{i}?", "response": " ".join(["word"] * 10)})
        for i in range(n)
    )


def test_generate_project_instruction_only():
    files = generate_project(
        domain="legal", model_key="qwen2.5-0.5b", instruction_data=_instruction_data()
    )
    assert "data/instruction_dataset.jsonl" in files
    assert "notebooks/instruction_finetuning.ipynb" in files
    assert "notebooks/non_instruction_finetuning.ipynb" not in files
    assert "notebooks/dpo_alignment.ipynb" not in files
    assert "README.md" in files


def test_generate_project_with_raw_text_includes_stage1():
    raw_text = "\n\n".join([" ".join(["word"] * 15)] * 10)
    files = generate_project(
        domain="legal",
        model_key="qwen2.5-0.5b",
        instruction_data=_instruction_data(),
        raw_text=raw_text,
    )
    assert "notebooks/non_instruction_finetuning.ipynb" in files
    assert "data/non_instruction_data.txt" in files


def test_generate_project_raises_on_invalid_instruction_data():
    try:
        generate_project(domain="legal", model_key="qwen2.5-0.5b", instruction_data="not jsonl")
        assert False, "expected ValidationFailed"
    except ValidationFailed as e:
        assert len(e.errors) > 0


def test_generate_project_raises_on_unknown_model():
    try:
        generate_project(
            domain="legal", model_key="does-not-exist", instruction_data=_instruction_data()
        )
        assert False, "expected ValidationFailed"
    except ValidationFailed as e:
        assert any("Unknown model" in err for err in e.errors)
