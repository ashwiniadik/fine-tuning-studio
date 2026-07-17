"""FastAPI app: upload dataset(s), get back a generated fine-tuning project zip."""

import io
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.generator import ValidationFailed, build_zip, generate_project
from backend.models_config import MODELS

app = FastAPI(title="Fine-Tuning Studio")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _decode_upload(filename: str, content: bytes) -> tuple[str | None, str | None]:
    """Decode uploaded bytes as UTF-8 text, returning (text, error_message)."""
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, f"{filename}: file is not valid UTF-8 text"


@app.get("/api/models")
def list_models():
    return {key: cfg["display_name"] for key, cfg in MODELS.items()}


@app.post("/api/generate")
async def generate(
    domain: str = Form(...),
    model: str = Form(...),
    instruction_data: UploadFile = File(...),
    raw_text: UploadFile | None = File(None),
    preference_data: UploadFile | None = File(None),
):
    instruction_content, instruction_error = _decode_upload(
        instruction_data.filename or "instruction_data", await instruction_data.read()
    )
    raw_text_content = raw_text_error = None
    if raw_text:
        raw_text_content, raw_text_error = _decode_upload(raw_text.filename or "raw_text", await raw_text.read())
    preference_content = preference_error = None
    if preference_data:
        preference_content, preference_error = _decode_upload(
            preference_data.filename or "preference_data", await preference_data.read()
        )

    decode_errors = [e for e in (instruction_error, raw_text_error, preference_error) if e]
    if decode_errors:
        raise HTTPException(status_code=400, detail=decode_errors)

    try:
        files = generate_project(
            domain=domain,
            model_key=model,
            instruction_data=instruction_content,
            raw_text=raw_text_content,
            preference_data=preference_content,
        )
    except ValidationFailed as e:
        raise HTTPException(status_code=400, detail=e.errors)

    zip_bytes = build_zip(files)
    safe_domain = re.sub(r"[^A-Za-z0-9._-]", "_", domain.strip()) or "project"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_domain}_finetuning.zip"'},
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
