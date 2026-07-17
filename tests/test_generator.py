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


def test_generate_project_raises_clean_error_on_domain_with_triple_quote():
    # A domain containing `"""` would otherwise produce a generated notebook
    # cell that fails to compile (SyntaxError: unterminated triple-quoted
    # string literal) when opened in Colab. It must be caught here as a clean
    # validation error instead.
    try:
        generate_project(
            domain='legal"""', model_key="qwen2.5-0.5b", instruction_data=_instruction_data()
        )
        assert False, "expected ValidationFailed"
    except ValidationFailed as e:
        assert any("domain" in err.lower() for err in e.errors)


def test_generate_project_raises_clean_error_on_domain_with_braces():
    # A domain containing `{` / `}` compiles fine but raises IndexError at
    # ALPACA_PROMPT.format() runtime in Colab. It must be caught here as a
    # clean validation error instead.
    try:
        generate_project(
            domain="legal {} braces", model_key="qwen2.5-0.5b", instruction_data=_instruction_data()
        )
        assert False, "expected ValidationFailed"
    except ValidationFailed as e:
        assert any("domain" in err.lower() for err in e.errors)


def test_generate_project_with_multiword_hyphenated_domain_succeeds():
    # Reasonable real-world domain names (multi-word, hyphenated) must not be
    # rejected by the new validation.
    files = generate_project(
        domain="e-commerce", model_key="qwen2.5-0.5b", instruction_data=_instruction_data()
    )
    assert "notebooks/instruction_finetuning.ipynb" in files


import io
import zipfile

from backend.generator import build_readme, build_zip


def test_build_zip_contains_all_files():
    files = {"a.txt": b"hello", "b/c.txt": b"world"}
    zip_bytes = build_zip(files)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    assert set(zf.namelist()) == {"a.txt", "b/c.txt"}
    assert zf.read("a.txt") == b"hello"
    assert zf.read("b/c.txt") == b"world"


def test_build_readme_lists_stages_in_order():
    readme = build_readme("legal", "Qwen2.5 0.5B", ["stage2", "stage3"])
    assert "1. `notebooks/instruction_finetuning.ipynb`" in readme
    assert "2. `notebooks/dpo_alignment.ipynb`" in readme
    assert "non_instruction_finetuning" not in readme


def test_build_readme_all_three_stages():
    readme = build_readme("legal", "Qwen2.5 0.5B", ["stage1", "stage2", "stage3"])
    assert "1. `notebooks/non_instruction_finetuning.ipynb`" in readme
    assert "2. `notebooks/instruction_finetuning.ipynb`" in readme
    assert "3. `notebooks/dpo_alignment.ipynb`" in readme
