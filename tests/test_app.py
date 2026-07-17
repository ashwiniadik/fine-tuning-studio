import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import FRONTEND_DIR, app

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


def test_generate_with_crlf_in_domain_does_not_crash():
    # A domain containing CRLF must not reach the Content-Disposition header
    # unescaped (would raise RuntimeError / crash a real ASGI server).
    response = client.post(
        "/api/generate",
        data={"domain": "legal\r\nX-Injected: evil", "model": "qwen2.5-0.5b"},
        files={
            "instruction_data": (
                "instruction_dataset.jsonl",
                _instruction_data(),
                "application/json",
            )
        },
    )
    assert response.status_code == 200
    content_disposition = response.headers["content-disposition"]
    assert "\r" not in content_disposition
    assert "\n" not in content_disposition


def test_generate_with_quote_in_domain_does_not_crash():
    # A domain containing a double quote must not break out of the quoted
    # filename value in the Content-Disposition header.
    response = client.post(
        "/api/generate",
        data={"domain": 'legal"; evil="x', "model": "qwen2.5-0.5b"},
        files={
            "instruction_data": (
                "instruction_dataset.jsonl",
                _instruction_data(),
                "application/json",
            )
        },
    )
    assert response.status_code == 200
    content_disposition = response.headers["content-disposition"]
    # Only the two quotes that legitimately wrap the filename value should remain.
    assert content_disposition.count('"') == 2


def test_generate_with_non_utf8_instruction_data_returns_400():
    response = client.post(
        "/api/generate",
        data={"domain": "legal", "model": "qwen2.5-0.5b"},
        files={
            "instruction_data": (
                "instruction_dataset.jsonl",
                b"\xff\xfe not valid utf-8 \x80\x81",
                "application/json",
            )
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert len(detail) > 0
    assert any("instruction_dataset.jsonl" in message and "UTF-8" in message for message in detail)


def test_frontend_dir_is_absolute():
    # StaticFiles must be mounted with a path resolved from the module
    # location, not the process's current working directory.
    assert Path(FRONTEND_DIR).is_absolute()
