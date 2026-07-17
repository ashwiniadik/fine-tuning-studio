"""FastAPI app: upload dataset(s), get back a generated fine-tuning project zip."""

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.generator import ValidationFailed, build_zip, generate_project
from backend.models_config import MODELS

app = FastAPI(title="Fine-Tuning Studio")


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
    instruction_content = (await instruction_data.read()).decode("utf-8")
    raw_text_content = (await raw_text.read()).decode("utf-8") if raw_text else None
    preference_content = (await preference_data.read()).decode("utf-8") if preference_data else None

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
    safe_domain = domain.strip().replace(" ", "_") or "project"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_domain}_finetuning.zip"'},
    )


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
