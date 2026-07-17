"""Exercises the full upload -> validate -> generate -> zip flow through the
real API for all 4 valid stage combinations, and checks that every generated
notebook is well-formed JSON with the expected structure -- catching anything
Task 12's narrower tests didn't cover."""

import io
import json
import zipfile

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def _instruction_data(n=20):
    return "\n".join(
        json.dumps({"instruction": f"Q{i}?", "response": " ".join(["word"] * 10)})
        for i in range(n)
    )


def _raw_text(n=10):
    return "\n\n".join([" ".join(["word"] * 15)] * n)


def _preference_data(n=20):
    return "\n".join(
        json.dumps({"prompt": f"Q{i}?", "chosen": "A specific answer.", "rejected": "A vague one."})
        for i in range(n)
    )


def _generate(domain, model, instruction=True, raw_text=False, preference=False):
    files = {}
    if instruction:
        files["instruction_data"] = ("instruction_dataset.jsonl", _instruction_data(), "application/json")
    if raw_text:
        files["raw_text"] = ("non_instruction_data.txt", _raw_text(), "text/plain")
    if preference:
        files["preference_data"] = ("preference_dataset.jsonl", _preference_data(), "application/json")
    return client.post("/api/generate", data={"domain": domain, "model": model}, files=files)


def _assert_notebook_valid(zf, path):
    nb = json.loads(zf.read(path))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 0
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")


def test_instruction_only_generates_sft_notebook_only():
    response = _generate("legal", "qwen2.5-0.5b")
    assert response.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = zf.namelist()
    assert names.count("notebooks/instruction_finetuning.ipynb") == 1
    assert "notebooks/non_instruction_finetuning.ipynb" not in names
    assert "notebooks/dpo_alignment.ipynb" not in names
    _assert_notebook_valid(zf, "notebooks/instruction_finetuning.ipynb")


def test_raw_text_and_instruction_generates_two_notebooks():
    response = _generate("legal", "qwen2.5-0.5b", raw_text=True)
    assert response.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = zf.namelist()
    assert "notebooks/non_instruction_finetuning.ipynb" in names
    assert "notebooks/instruction_finetuning.ipynb" in names
    assert "notebooks/dpo_alignment.ipynb" not in names
    _assert_notebook_valid(zf, "notebooks/non_instruction_finetuning.ipynb")
    _assert_notebook_valid(zf, "notebooks/instruction_finetuning.ipynb")


def test_instruction_and_preference_generates_two_notebooks():
    response = _generate("legal", "qwen2.5-0.5b", preference=True)
    assert response.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = zf.namelist()
    assert "notebooks/non_instruction_finetuning.ipynb" not in names
    assert "notebooks/instruction_finetuning.ipynb" in names
    assert "notebooks/dpo_alignment.ipynb" in names
    _assert_notebook_valid(zf, "notebooks/instruction_finetuning.ipynb")
    _assert_notebook_valid(zf, "notebooks/dpo_alignment.ipynb")


def test_all_three_files_generates_full_pipeline():
    response = _generate("legal", "qwen2.5-0.5b", raw_text=True, preference=True)
    assert response.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = zf.namelist()
    for path in [
        "notebooks/non_instruction_finetuning.ipynb",
        "notebooks/instruction_finetuning.ipynb",
        "notebooks/dpo_alignment.ipynb",
        "data/non_instruction_data.txt",
        "data/instruction_dataset.jsonl",
        "data/preference_dataset.jsonl",
        "README.md",
    ]:
        assert path in names, path
        if path.endswith(".ipynb"):
            _assert_notebook_valid(zf, path)


def test_every_curated_model_generates_successfully():
    from backend.models_config import MODELS

    for model_key in MODELS:
        response = _generate("legal", model_key, raw_text=True, preference=True)
        assert response.status_code == 200, f"{model_key}: {response.text}"
