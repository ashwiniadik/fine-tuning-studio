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


def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "qwen2.5-0.5b" in response.json()


def test_generate_with_only_instruction_data_returns_zip():
    response = client.post(
        "/api/generate",
        data={"domain": "legal", "model": "qwen2.5-0.5b"},
        files={
            "instruction_data": (
                "instruction_dataset.jsonl",
                _instruction_data(),
                "application/json",
            )
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = zf.namelist()
    assert "data/instruction_dataset.jsonl" in names
    assert "notebooks/instruction_finetuning.ipynb" in names
    assert "notebooks/non_instruction_finetuning.ipynb" not in names


def test_generate_with_invalid_data_returns_400():
    response = client.post(
        "/api/generate",
        data={"domain": "legal", "model": "qwen2.5-0.5b"},
        files={"instruction_data": ("instruction_dataset.jsonl", "not valid jsonl", "application/json")},
    )
    assert response.status_code == 400
    assert len(response.json()["detail"]) > 0
