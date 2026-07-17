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


def test_generate_with_browser_style_empty_optional_files_returns_zip():
    # A real browser's `new FormData(form)` always includes an entry for every
    # <input type="file">, even ones the user never touched -- as an empty
    # File object with an *empty but present* filename (`filename=""` in the
    # multipart Content-Disposition header), not an absent field. UploadFile
    # has no __bool__, so `if raw_text:` is always truthy for such a field;
    # only checking `.filename` distinguishes "no file chosen" from
    # "provided". Without that check this exact request 400s in a real
    # browser even though the user only meant to fill in the required
    # instruction dataset -- confirmed against a live server via Playwright.
    #
    # httpx's `files={"raw_text": ("", b"", ...)}` convenience form does NOT
    # reproduce this: httpx silently omits the `filename` attribute entirely
    # when it's an empty string, which is a different wire format than what
    # browsers send and does not trigger the bug. The multipart body must be
    # built by hand to include `filename=""` explicitly, matching real
    # browser output.
    instruction_data = _instruction_data()
    boundary = "browserstyleboundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="domain"\r\n\r\n'
        "legal\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "qwen2.5-0.5b\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="instruction_data"; filename="instruction_dataset.jsonl"\r\n'
        "Content-Type: application/json\r\n\r\n"
        f"{instruction_data}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="raw_text"; filename=""\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
        "\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="preference_data"; filename=""\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
        "\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    response = client.post(
        "/api/generate",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    names = zf.namelist()
    assert "notebooks/non_instruction_finetuning.ipynb" not in names
    assert "notebooks/dpo_alignment.ipynb" not in names


def test_generate_with_invalid_data_returns_400():
    response = client.post(
        "/api/generate",
        data={"domain": "legal", "model": "qwen2.5-0.5b"},
        files={"instruction_data": ("instruction_dataset.jsonl", "not valid jsonl", "application/json")},
    )
    assert response.status_code == 400
    assert len(response.json()["detail"]) > 0


def test_generate_with_crlf_in_domain_returns_clean_400():
    # A domain containing CRLF is now rejected by domain validation before it
    # ever reaches the Content-Disposition header or a generated notebook --
    # a clean 400, not a crash and not a 200 with injected bytes smuggled
    # through.
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
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("domain" in message.lower() for message in detail)


def test_generate_with_quote_in_domain_returns_clean_400():
    # A domain containing a double quote is now rejected by domain validation
    # -- both because it would break out of the quoted Content-Disposition
    # filename value, and because it would break the triple-quoted Python
    # string literal the domain gets embedded into in generated notebooks.
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
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("domain" in message.lower() for message in detail)


def test_generate_with_braces_in_domain_returns_clean_400():
    # A domain containing `{` / `}` compiles fine as embedded Python source
    # but breaks the generated notebook's ALPACA_PROMPT.format() call at
    # runtime in Colab (IndexError: Replacement index out of range). This
    # must be caught at validation time instead of shipping a notebook that
    # crashes when the user actually runs it.
    response = client.post(
        "/api/generate",
        data={"domain": "legal {} braces", "model": "qwen2.5-0.5b"},
        files={
            "instruction_data": (
                "instruction_dataset.jsonl",
                _instruction_data(),
                "application/json",
            )
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("domain" in message.lower() for message in detail)


def test_generate_with_triple_quote_in_domain_returns_clean_400():
    # A domain containing `"""` would otherwise produce a generated notebook
    # code cell that fails to even compile() (SyntaxError: unterminated
    # triple-quoted string literal) when opened in Colab.
    response = client.post(
        "/api/generate",
        data={"domain": 'legal"""', "model": "qwen2.5-0.5b"},
        files={
            "instruction_data": (
                "instruction_dataset.jsonl",
                _instruction_data(),
                "application/json",
            )
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("domain" in message.lower() for message in detail)


def test_generate_with_multiword_hyphenated_domain_returns_zip():
    # Real, reasonable domain names (multi-word, hyphenated) must still work
    # -- the validator's job is to block syntax-breaking characters, not to
    # restrict domains to single words.
    for domain in ("customer support", "e-commerce", "K-12 education"):
        response = client.post(
            "/api/generate",
            data={"domain": domain, "model": "qwen2.5-0.5b"},
            files={
                "instruction_data": (
                    "instruction_dataset.jsonl",
                    _instruction_data(),
                    "application/json",
                )
            },
        )
        assert response.status_code == 200, f"domain {domain!r} unexpectedly rejected: {response.text}"


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
