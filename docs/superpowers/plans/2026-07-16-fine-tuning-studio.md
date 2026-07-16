# Fine-Tuning Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + plain-HTML web tool that generates ready-to-run Colab notebooks for fine-tuning a curated small model on a user's own uploaded dataset, for any domain — generalizing the Unsloth pipeline already built and debugged in `finance-faq-assistant-finetuning`.

**Architecture:** A stateless backend validates uploaded dataset files, selects which of the 3 pipeline stages apply based on which files were provided, fills parameterized notebook templates (ported from the Finance project's already-fixed code), and returns a downloadable zip. No GPU, database, or job queue anywhere in this system — actual training still runs on the user's own free Colab session, same as before.

**Tech Stack:** Python, FastAPI, `pytest` + FastAPI `TestClient`, plain HTML/CSS/JS (no build step). Unlike the Finance project, every part of this system is testable and runnable locally without a GPU.

---

## File Structure

```
fine-tuning-studio/
├── backend/
│   ├── __init__.py
│   ├── app.py                    # Task 12
│   ├── validators.py             # Tasks 2-4
│   ├── models_config.py          # Task 5
│   ├── generator.py              # Tasks 9-11
│   └── notebook_templates/
│       ├── __init__.py           # Task 6
│       ├── _helpers.py           # Task 6
│       ├── stage1.py             # Task 6
│       ├── stage2.py             # Task 7
│       └── stage3.py             # Task 8
├── frontend/
│   ├── index.html                # Task 13
│   ├── app.js                    # Task 13
│   └── style.css                 # Task 13
├── tests/
│   ├── __init__.py
│   ├── test_validators.py        # Tasks 2-4
│   ├── test_models_config.py     # Task 5
│   ├── test_notebook_templates.py # Tasks 6-8
│   ├── test_generator.py         # Tasks 9-11
│   ├── test_app.py               # Task 12
│   └── test_integration.py       # Task 15
├── README.md                     # Task 14
├── requirements.txt               # Task 1
└── pytest.ini                     # Task 1
```

---

### Task 1: Project skeleton & tooling

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `backend/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p backend/notebook_templates frontend tests
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi
uvicorn[standard]
python-multipart
httpx
pytest
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 4: Create empty package markers**

```bash
touch backend/__init__.py backend/notebook_templates/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install dependencies and verify pytest runs**

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest --collect-only
```

Expected: `collected 0 items`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini backend/__init__.py backend/notebook_templates/__init__.py tests/__init__.py
git commit -m "chore: project skeleton and pytest setup"
```

---

### Task 2: Raw text validator

**Files:**
- Create: `backend/validators.py`
- Create: `tests/test_validators.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validators.py`:

```python
from backend.validators import validate_raw_text


def _make_raw_text(n_paragraphs=10, words_per_paragraph=15):
    paragraph = " ".join(["word"] * words_per_paragraph)
    return "\n\n".join([paragraph] * n_paragraphs)


def test_valid_raw_text_returns_no_errors():
    assert validate_raw_text(_make_raw_text()) == []


def test_raw_text_too_few_paragraphs():
    errors = validate_raw_text(_make_raw_text(n_paragraphs=5))
    assert any("at least 10 paragraphs" in e for e in errors)


def test_raw_text_paragraph_too_short():
    text = _make_raw_text(n_paragraphs=9) + "\n\nToo short."
    errors = validate_raw_text(text)
    assert any("fewer than 15 words" in e for e in errors)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_validators.py -v
```

Expected: FAIL — `backend/validators.py` (and `validate_raw_text`) does not exist.

- [ ] **Step 3: Write `backend/validators.py`**

```python
"""Generalized dataset validators for Fine-Tuning Studio.

Looser than finance-faq-assistant-finetuning's assignment-specific minimums
(50+/100+/50+), since this tool supports arbitrary domains, not one fixed
assignment. Note: unlike that project's tests, the preference validator here
deliberately does NOT require `chosen` to be longer than `rejected` -- that
rule was found during the Finance project's code review to be a length-confound
anti-pattern that risks teaching DPO to prefer verbosity over correctness.
"""

MIN_RAW_TEXT_PARAGRAPHS = 10
MIN_PARAGRAPH_WORDS = 15


def validate_raw_text(content: str) -> list[str]:
    errors = []
    paragraphs = [p.strip() for p in content.strip().split("\n\n") if p.strip()]
    if len(paragraphs) < MIN_RAW_TEXT_PARAGRAPHS:
        errors.append(
            f"Raw text file needs at least {MIN_RAW_TEXT_PARAGRAPHS} paragraphs, "
            f"found {len(paragraphs)}."
        )
    short = [p for p in paragraphs if len(p.split()) < MIN_PARAGRAPH_WORDS]
    if short:
        errors.append(
            f"{len(short)} paragraph(s) have fewer than {MIN_PARAGRAPH_WORDS} words."
        )
    return errors
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
python3 -m pytest tests/test_validators.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/validators.py tests/test_validators.py
git commit -m "feat: add raw text dataset validator"
```

---

### Task 3: Instruction JSONL validator

**Files:**
- Modify: `backend/validators.py` (append `_parse_jsonl`, `validate_instruction_jsonl`)
- Modify: `tests/test_validators.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validators.py`:

```python
import json

from backend.validators import validate_instruction_jsonl


def _make_instruction_jsonl(n=20, response_words=10):
    lines = []
    for i in range(n):
        response = " ".join(["word"] * response_words)
        lines.append(json.dumps({"instruction": f"Question {i}?", "response": response}))
    return "\n".join(lines)


def test_valid_instruction_jsonl_returns_no_errors():
    assert validate_instruction_jsonl(_make_instruction_jsonl()) == []


def test_instruction_jsonl_too_few_examples():
    errors = validate_instruction_jsonl(_make_instruction_jsonl(n=5))
    assert any("at least 20 examples" in e for e in errors)


def test_instruction_jsonl_bad_schema():
    content = json.dumps({"instruction": "Q?", "answer": "wrong key name"})
    errors = validate_instruction_jsonl(content)
    assert any("exactly 'instruction' and 'response'" in e for e in errors)


def test_instruction_jsonl_short_response():
    content = json.dumps({"instruction": "Q?", "response": "too short"})
    errors = validate_instruction_jsonl(content)
    assert any("fewer than 5 words" in e for e in errors)


def test_instruction_jsonl_invalid_json_line():
    errors = validate_instruction_jsonl("not valid json")
    assert any("not valid JSON" in e for e in errors)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_validators.py -v -k instruction_jsonl
```

Expected: FAIL — `validate_instruction_jsonl` does not exist.

- [ ] **Step 3: Append to `backend/validators.py`**

```python
import json


def _parse_jsonl(content: str) -> tuple[list[dict], list[str]]:
    records = []
    errors = []
    for i, line in enumerate(content.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"Line {i + 1} is not valid JSON: {e}")
    return records, errors


MIN_INSTRUCTION_EXAMPLES = 20
MIN_RESPONSE_WORDS = 5


def validate_instruction_jsonl(content: str) -> list[str]:
    errors = []
    records, parse_errors = _parse_jsonl(content)
    errors.extend(parse_errors)
    if len(records) < MIN_INSTRUCTION_EXAMPLES:
        errors.append(
            f"Instruction dataset needs at least {MIN_INSTRUCTION_EXAMPLES} examples, "
            f"found {len(records)}."
        )
    for i, record in enumerate(records):
        if set(record.keys()) != {"instruction", "response"}:
            errors.append(
                f"Record {i} must have exactly 'instruction' and 'response' keys, "
                f"got {sorted(record.keys())}."
            )
            continue
        if not isinstance(record["instruction"], str) or not record["instruction"].strip():
            errors.append(f"Record {i} has an empty instruction.")
        if (
            not isinstance(record["response"], str)
            or len(record["response"].split()) < MIN_RESPONSE_WORDS
        ):
            errors.append(f"Record {i} has a response with fewer than {MIN_RESPONSE_WORDS} words.")
    return errors
```

Add the `import json` line at the top of the file (next to any existing imports) rather than duplicating it if a lower step already needs it.

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
python3 -m pytest tests/test_validators.py -v
```

Expected: all 8 tests PASS (3 from Task 2 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add backend/validators.py tests/test_validators.py
git commit -m "feat: add instruction dataset validator"
```

---

### Task 4: Preference JSONL validator

**Files:**
- Modify: `backend/validators.py` (append `validate_preference_jsonl`)
- Modify: `tests/test_validators.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validators.py`:

```python
from backend.validators import validate_preference_jsonl


def _make_preference_jsonl(n=20):
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "prompt": f"Question {i}?",
                    "chosen": "A specific, correct, helpful answer.",
                    "rejected": "A vague non-answer.",
                }
            )
        )
    return "\n".join(lines)


def test_valid_preference_jsonl_returns_no_errors():
    assert validate_preference_jsonl(_make_preference_jsonl()) == []


def test_preference_jsonl_too_few_examples():
    errors = validate_preference_jsonl(_make_preference_jsonl(n=5))
    assert any("at least 20 examples" in e for e in errors)


def test_preference_jsonl_bad_schema():
    content = json.dumps({"prompt": "Q?", "chosen": "A", "rejected": "B", "extra": "C"})
    errors = validate_preference_jsonl(content)
    assert any("exactly 'prompt', 'chosen', and 'rejected'" in e for e in errors)


def test_preference_jsonl_identical_chosen_and_rejected():
    content = json.dumps({"prompt": "Q?", "chosen": "Same text.", "rejected": "Same text."})
    errors = validate_preference_jsonl(content)
    assert any("identical chosen and rejected" in e for e in errors)


def test_preference_jsonl_does_not_require_chosen_longer_than_rejected():
    # Deliberate design decision: the Finance project's tests required chosen
    # to be a longer word count than rejected, but that rule was flagged during
    # that project's code review as a length-confound anti-pattern for DPO
    # training. This validator intentionally does not enforce it -- a short,
    # sharp chosen answer must not be rejected by the validator just for being
    # shorter than a padded-out rejected one.
    content = json.dumps(
        {
            "prompt": "What is a SIP?",
            "chosen": "A SIP is a plan.",
            "rejected": (
                "A systematic investment plan lets you invest a fixed amount "
                "regularly into a mutual fund over an extended period of time."
            ),
        }
    )
    errors = validate_preference_jsonl(content)
    assert not any("longer" in e.lower() for e in errors)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_validators.py -v -k preference_jsonl
```

Expected: FAIL — `validate_preference_jsonl` does not exist.

- [ ] **Step 3: Append to `backend/validators.py`**

```python
MIN_PREFERENCE_EXAMPLES = 20


def validate_preference_jsonl(content: str) -> list[str]:
    errors = []
    records, parse_errors = _parse_jsonl(content)
    errors.extend(parse_errors)
    if len(records) < MIN_PREFERENCE_EXAMPLES:
        errors.append(
            f"Preference dataset needs at least {MIN_PREFERENCE_EXAMPLES} examples, "
            f"found {len(records)}."
        )
    for i, record in enumerate(records):
        if set(record.keys()) != {"prompt", "chosen", "rejected"}:
            errors.append(
                f"Record {i} must have exactly 'prompt', 'chosen', and 'rejected' keys, "
                f"got {sorted(record.keys())}."
            )
            continue
        if not record["chosen"].strip() or not record["rejected"].strip():
            errors.append(f"Record {i} has an empty chosen or rejected response.")
        if record["chosen"] == record["rejected"]:
            errors.append(f"Record {i} has identical chosen and rejected responses.")
    return errors
```

- [ ] **Step 4: Run the full validator test suite**

```bash
python3 -m pytest tests/test_validators.py -v
```

Expected: all 13 tests PASS (8 from Tasks 2-3 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add backend/validators.py tests/test_validators.py
git commit -m "feat: add preference dataset validator (no length-confound rule)"
```

---

### Task 5: Model configuration

**Files:**
- Create: `backend/models_config.py`
- Create: `tests/test_models_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_config.py`:

```python
from backend.models_config import MODELS

REQUIRED_STAGE_KEYS = {"learning_rate", "epochs"}
REQUIRED_LORA_KEYS = {"r", "lora_alpha", "lora_dropout", "target_modules"}


def test_expected_models_present():
    assert set(MODELS.keys()) == {
        "qwen2.5-0.5b",
        "qwen2.5-1.5b",
        "tinyllama-1.1b",
        "llama-3.2-1b",
    }


def test_all_models_have_required_fields():
    for key, cfg in MODELS.items():
        assert "display_name" in cfg, key
        assert cfg["unsloth_model_id"].startswith("unsloth/"), key
        assert set(cfg["lora"].keys()) == REQUIRED_LORA_KEYS, key
        for stage in ["stage1", "stage2", "stage3"]:
            assert stage in cfg, f"{key} missing {stage}"
            assert set(cfg[stage].keys()) == REQUIRED_STAGE_KEYS, f"{key} {stage}"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python3 -m pytest tests/test_models_config.py -v
```

Expected: FAIL — `backend/models_config.py` does not exist.

- [ ] **Step 3: Write `backend/models_config.py`**

Model IDs verified directly against Hugging Face (`unsloth/Qwen2.5-0.5B`, `unsloth/Qwen2.5-1.5B`, `unsloth/tinyllama-bnb-4bit`, `unsloth/Llama-3.2-1B` all confirmed to exist as base, non-instruct models). LoRA target modules are the same 7-module attention+MLP list validated in the Finance project; this is expected to hold across all four models since they share standard Llama-family module naming, but should be spot-checked once notebooks are actually generated and run for each model.

```python
"""Curated base models and their fine-tuning defaults for Fine-Tuning Studio.

Hyperparameter defaults (LoRA config, learning rates, epochs) reuse the exact
values validated in finance-faq-assistant-finetuning's three notebooks.
"""

DEFAULT_LORA = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

DEFAULT_STAGE_HPARAMS = {
    "stage1": {"learning_rate": 2e-4, "epochs": 2},
    "stage2": {"learning_rate": 2e-4, "epochs": 3},
    "stage3": {"learning_rate": 5e-6, "epochs": 2},
}

MODELS = {
    "qwen2.5-0.5b": {
        "display_name": "Qwen2.5 0.5B",
        "unsloth_model_id": "unsloth/Qwen2.5-0.5B",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
    "qwen2.5-1.5b": {
        "display_name": "Qwen2.5 1.5B",
        "unsloth_model_id": "unsloth/Qwen2.5-1.5B",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
    "tinyllama-1.1b": {
        "display_name": "TinyLlama 1.1B",
        "unsloth_model_id": "unsloth/tinyllama-bnb-4bit",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
    "llama-3.2-1b": {
        "display_name": "Llama 3.2 1B",
        "unsloth_model_id": "unsloth/Llama-3.2-1B",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
}
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m pytest tests/test_models_config.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/models_config.py tests/test_models_config.py
git commit -m "feat: add curated model configuration"
```

---

### Task 6: Notebook template helpers + Stage 1 template

**Files:**
- Create: `backend/notebook_templates/_helpers.py`
- Create: `backend/notebook_templates/stage1.py`
- Create: `tests/test_notebook_templates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_notebook_templates.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python3 -m pytest tests/test_notebook_templates.py -v
```

Expected: FAIL — `backend/notebook_templates/stage1.py` does not exist.

- [ ] **Step 3: Write `backend/notebook_templates/_helpers.py`**

```python
"""Shared cell-building helpers for notebook templates."""


def markdown(source: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: list[str]) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def lora_target_modules_literal(target_modules: list[str]) -> str:
    return ", ".join(f'"{m}"' for m in target_modules)


def alpaca_prompt_text(domain: str) -> str:
    """The exact Alpaca-style prompt template text, shared by every template
    that needs it (Stage 2 SFT and Stage 3 DPO), so both are guaranteed to use
    an identical string for a given domain rather than two hand-written copies
    that could silently drift apart."""
    return (
        f"Below is an instruction that describes a {domain} question. "
        "Write a response that answers it accurately and clearly.\n"
        "\n"
        "### Instruction:\n"
        "{}\n"
        "\n"
        "### Response:\n"
        "{}"
    )
```

- [ ] **Step 4: Write `backend/notebook_templates/stage1.py`**

```python
"""Stage 1 (non-instruction fine-tuning) notebook template."""

from backend.notebook_templates._helpers import code, lora_target_modules_literal, markdown, notebook


def build_stage1_notebook(domain: str, model_id: str, lora: dict, learning_rate: float, epochs: int) -> dict:
    target_modules = lora_target_modules_literal(lora["target_modules"])

    cells = [
        markdown([
            "# Stage 1: Non-Instruction Fine-Tuning\n",
            "\n",
            f"This notebook adapts `{model_id}` to {domain} vocabulary and tone by "
            "training directly on raw domain text (no question/answer structure yet "
            "-- that comes in Stage 2). Run this on Google Colab with a T4 GPU: "
            "Runtime > Change runtime type > T4 GPU.",
        ]),
        code([
            "%%capture\n",
            "!pip install unsloth\n",
            "!pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git",
        ]),
        markdown([
            "## 1. Load the raw domain text\n",
            "\n",
            "Upload `data/non_instruction_data.txt` to this Colab session (or clone "
            "the repo), then read it in. Paragraphs are separated by blank lines.",
        ]),
        code([
            'raw_text = open("data/non_instruction_data.txt", encoding="utf-8").read()\n',
            'paragraphs = [p.strip() for p in raw_text.split("\\n\\n") if p.strip()]\n',
            'print(f"Loaded {len(paragraphs)} paragraphs")\n',
            "print(paragraphs[0])",
        ]),
        markdown([
            "## 2. Clean and chunk the text\n",
            "\n",
            "Each paragraph is already a short, self-contained chunk, so no further "
            "splitting is needed. We just wrap each one into a HuggingFace `Dataset` "
            "with a single `text` column.",
        ]),
        code([
            "from datasets import Dataset\n",
            "\n",
            'dataset = Dataset.from_dict({"text": paragraphs})\n',
            "dataset",
        ]),
        markdown(["## 3. Load the base model using Unsloth"]),
        code([
            "from unsloth import FastLanguageModel\n",
            "import torch\n",
            "\n",
            "max_seq_length = 2048\n",
            "\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{model_id}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")",
        ]),
        markdown([
            "## 4. Apply LoRA (QLoRA, since the base model is loaded in 4-bit)\n",
            "\n",
            "`r` is the LoRA rank -- how large the small trainable adapter matrices "
            "are. `lora_alpha` scales how strongly the adapter's updates affect the "
            "model. `target_modules` lists which layers get an adapter -- here, all "
            "the attention and MLP projection layers.",
        ]),
        code([
            "model = FastLanguageModel.get_peft_model(\n",
            "    model,\n",
            f'    r = {lora["r"]},\n',
            f"    target_modules = [{target_modules}],\n",
            f'    lora_alpha = {lora["lora_alpha"]},\n',
            f'    lora_dropout = {lora["lora_dropout"]},\n',
            '    bias = "none",\n',
            '    use_gradient_checkpointing = "unsloth",\n',
            "    random_state = 3407,\n",
            ")",
        ]),
        markdown([
            "## 5. Train on the raw text\n",
            "\n",
            "This is plain next-token-prediction training on the `text` column -- "
            "the same objective used for the original pretraining, just on our "
            f"{domain} paragraphs instead of the internet.",
        ]),
        code([
            "from trl import SFTTrainer, SFTConfig\n",
            "from unsloth import is_bfloat16_supported\n",
            "\n",
            "trainer = SFTTrainer(\n",
            "    model = model,\n",
            "    processing_class = tokenizer,\n",
            "    train_dataset = dataset,\n",
            "    args = SFTConfig(\n",
            '        dataset_text_field = "text",\n',
            "        max_length = max_seq_length,\n",
            "        packing = True,\n",
            "        per_device_train_batch_size = 2,\n",
            "        gradient_accumulation_steps = 4,\n",
            f"        num_train_epochs = {epochs},\n",
            f"        learning_rate = {learning_rate},\n",
            "        fp16 = not is_bfloat16_supported(),\n",
            "        bf16 = is_bfloat16_supported(),\n",
            "        logging_steps = 5,\n",
            '        output_dir = "stage1_outputs",\n',
            '        optim = "adamw_8bit",\n',
            "        seed = 3407,\n",
            "    ),\n",
            ")\n",
            "trainer_stats = trainer.train()",
        ]),
        markdown(["## 6. Save the adapter"]),
        code([
            'model.save_pretrained("stage1_adapter")\n',
            'tokenizer.save_pretrained("stage1_adapter")',
        ]),
        markdown([
            "## 7. Test the model after non-instruction fine-tuning\n",
            "\n",
            f"We only expect the model to sound more {domain}-flavored here, not "
            "yet answer questions well -- instruction-following comes in Stage 2.",
        ]),
        code([
            "FastLanguageModel.for_inference(model)\n",
            'inputs = tokenizer([paragraphs[0][:30]], return_tensors="pt").to("cuda")\n',
            "outputs = model.generate(**inputs, max_new_tokens=64)\n",
            'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        ]),
    ]
    return notebook(cells)
```

- [ ] **Step 5: Run the test to confirm it passes**

```bash
python3 -m pytest tests/test_notebook_templates.py -v
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/notebook_templates/_helpers.py backend/notebook_templates/stage1.py tests/test_notebook_templates.py
git commit -m "feat: add notebook template helpers and Stage 1 template"
```

---

### Task 7: Stage 2 (SFT) template

**Files:**
- Create: `backend/notebook_templates/stage2.py`
- Modify: `tests/test_notebook_templates.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_notebook_templates.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_notebook_templates.py -v -k stage2
```

Expected: FAIL — `backend/notebook_templates/stage2.py` does not exist.

- [ ] **Step 3: Write `backend/notebook_templates/stage2.py`**

```python
"""Stage 2 (instruction fine-tuning / SFT) notebook template."""

from backend.notebook_templates._helpers import alpaca_prompt_text, code, lora_target_modules_literal, markdown, notebook


def build_stage2_notebook(
    domain: str,
    model_id: str,
    lora: dict,
    learning_rate: float,
    epochs: int,
    previous_adapter: str | None,
) -> dict:
    target_modules = lora_target_modules_literal(lora["target_modules"])
    load_model_name = previous_adapter if previous_adapter else model_id
    prompt_text = alpaca_prompt_text(domain)

    intro = (
        f"Continues from the Stage 1 adapter and teaches the model to answer "
        f"{domain} questions directly, using the instruction/response dataset."
        if previous_adapter
        else f"Teaches `{model_id}` to answer {domain} questions directly, using "
        "the instruction/response dataset."
    )
    upload_note = (
        f"Upload the `{previous_adapter}` folder from Stage 1, plus "
        "`data/instruction_dataset.jsonl`."
        if previous_adapter
        else "Upload `data/instruction_dataset.jsonl` to this Colab session (or "
        "clone the repo)."
    )

    cells = [
        markdown([
            "# Stage 2: Instruction Fine-Tuning (SFT)\n",
            "\n",
            f"{intro} Run on Google Colab with a T4 GPU.",
        ]),
        code([
            "%%capture\n",
            "!pip install unsloth\n",
            "!pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git",
        ]),
        markdown([
            "## 1. Load tokenizer and the model\n",
            "\n",
            upload_note,
        ]),
        code([
            "from unsloth import FastLanguageModel\n",
            "import torch\n",
            "\n",
            "max_seq_length = 2048\n",
            "\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{load_model_name}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")",
        ]),
        markdown([
            "## 2. Load and format the instruction dataset\n",
            "\n",
            "Each `{instruction, response}` pair is wrapped in a simple Alpaca-style "
            "prompt template so the model learns the exact format it will be asked "
            "to answer in at inference time.",
        ]),
        code([
            "from datasets import load_dataset\n",
            "\n",
            f'ALPACA_PROMPT = """{prompt_text}"""\n',
            "\n",
            "EOS_TOKEN = tokenizer.eos_token\n",
            "\n",
            "def formatting_prompts_func(examples):\n",
            "    texts = []\n",
            '    for instruction, response in zip(examples["instruction"], examples["response"]):\n',
            "        texts.append(ALPACA_PROMPT.format(instruction, response) + EOS_TOKEN)\n",
            '    return {"text": texts}\n',
            "\n",
            'dataset = load_dataset("json", data_files="data/instruction_dataset.jsonl", split="train")\n',
            "dataset = dataset.map(formatting_prompts_func, batched=True)\n",
            'dataset[0]["text"]',
        ]),
        markdown(["## 3. Apply LoRA"]),
        code([
            "model = FastLanguageModel.get_peft_model(\n",
            "    model,\n",
            f'    r = {lora["r"]},\n',
            f"    target_modules = [{target_modules}],\n",
            f'    lora_alpha = {lora["lora_alpha"]},\n',
            f'    lora_dropout = {lora["lora_dropout"]},\n',
            '    bias = "none",\n',
            '    use_gradient_checkpointing = "unsloth",\n',
            "    random_state = 3407,\n",
            ")",
        ]),
        markdown([
            "## 4. Train the model\n",
            "\n",
            "`packing=False` here because each instruction/response pair needs to "
            "stay its own training example with a clean boundary -- packing "
            "multiple short examples into one sequence would blur those "
            "boundaries and give far fewer real training steps.",
        ]),
        code([
            "from trl import SFTTrainer, SFTConfig\n",
            "from unsloth import is_bfloat16_supported\n",
            "\n",
            "trainer = SFTTrainer(\n",
            "    model = model,\n",
            "    processing_class = tokenizer,\n",
            "    train_dataset = dataset,\n",
            "    args = SFTConfig(\n",
            '        dataset_text_field = "text",\n',
            "        max_length = max_seq_length,\n",
            "        packing = False,\n",
            "        per_device_train_batch_size = 2,\n",
            "        gradient_accumulation_steps = 4,\n",
            f"        num_train_epochs = {epochs},\n",
            f"        learning_rate = {learning_rate},\n",
            "        fp16 = not is_bfloat16_supported(),\n",
            "        bf16 = is_bfloat16_supported(),\n",
            "        logging_steps = 5,\n",
            '        output_dir = "stage2_outputs",\n',
            '        optim = "adamw_8bit",\n',
            "        seed = 3407,\n",
            "    ),\n",
            ")\n",
            "trainer_stats = trainer.train()",
        ]),
        markdown(["## 5. Save the adapter"]),
        code([
            'model.save_pretrained("stage2_adapter")\n',
            'tokenizer.save_pretrained("stage2_adapter")',
        ]),
        markdown(["## 6. Run inference after training"]),
        code([
            "import json as _json\n",
            "\n",
            'first_question = _json.loads(open("data/instruction_dataset.jsonl").readline())["instruction"]\n',
            "\n",
            "FastLanguageModel.for_inference(model)\n",
            'prompt = ALPACA_PROMPT.format(first_question, "")\n',
            'inputs = tokenizer([prompt], return_tensors="pt").to("cuda")\n',
            "outputs = model.generate(**inputs, max_new_tokens=128)\n",
            'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        ]),
    ]
    return notebook(cells)
```

- [ ] **Step 4: Run the full notebook template test suite**

```bash
python3 -m pytest tests/test_notebook_templates.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/notebook_templates/stage2.py tests/test_notebook_templates.py
git commit -m "feat: add Stage 2 (SFT) notebook template"
```

---

### Task 8: Stage 3 (DPO) template

**Files:**
- Create: `backend/notebook_templates/stage3.py`
- Modify: `tests/test_notebook_templates.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_notebook_templates.py`:

```python
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
    assert prompt_text in json.dumps(nb2)
    assert prompt_text in json.dumps(nb3)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_notebook_templates.py -v -k stage3
```

Expected: FAIL — `backend/notebook_templates/stage3.py` does not exist.

- [ ] **Step 3: Write `backend/notebook_templates/stage3.py`**

```python
"""Stage 3 (DPO preference alignment) notebook template."""

from backend.notebook_templates._helpers import alpaca_prompt_text, code, lora_target_modules_literal, markdown, notebook


def build_stage3_notebook(
    domain: str,
    model_id: str,
    lora: dict,
    learning_rate: float,
    epochs: int,
    previous_adapter: str,
) -> dict:
    target_modules = lora_target_modules_literal(lora["target_modules"])
    prompt_text = alpaca_prompt_text(domain)

    cells = [
        markdown([
            "# Stage 3: DPO Preference Alignment\n",
            "\n",
            f"Continues from the Stage 2 SFT adapter and uses the preference "
            f"dataset to teach the model to prefer specific, helpful {domain} "
            "answers over generic ones. Run on Google Colab with a T4 GPU.",
        ]),
        code([
            "%%capture\n",
            "!pip install unsloth trl\n",
            "!pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git",
        ]),
        markdown([
            "## 1. Load the SFT model\n",
            "\n",
            f"Upload the `{previous_adapter}` folder from Stage 2, plus "
            "`data/preference_dataset.jsonl`. We load the SFT adapter twice: once "
            "as the trainable model DPO will update, and once more as a frozen "
            "reference copy -- DPO measures how far the trainable model's "
            "preferences have shifted relative to this frozen SFT reference, so "
            "the reference must actually be the SFT model, not the original "
            "pretrained base model.",
        ]),
        code([
            "from unsloth import FastLanguageModel\n",
            "\n",
            "max_seq_length = 2048\n",
            "\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{previous_adapter}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")\n",
            "model = FastLanguageModel.get_peft_model(\n",
            "    model,\n",
            f'    r = {lora["r"]},\n',
            f"    target_modules = [{target_modules}],\n",
            f'    lora_alpha = {lora["lora_alpha"]},\n',
            f'    lora_dropout = {lora["lora_dropout"]},\n',
            '    bias = "none",\n',
            '    use_gradient_checkpointing = "unsloth",\n',
            "    random_state = 3407,\n",
            ")\n",
            "\n",
            "# A second, separate copy of the SFT model, loaded without a fresh\n",
            "# LoRA adapter attached, so it stays frozen and serves as the DPO reference.\n",
            "ref_model, _ = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{previous_adapter}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")",
        ]),
        markdown([
            "## 2. Load and format the preference dataset\n",
            "\n",
            "The file already has `prompt`, `chosen`, and `rejected` columns, "
            "which is exactly what `DPOTrainer` expects. The `chosen`/`rejected` "
            "responses can be used as-is, but `prompt` needs to be wrapped in the "
            "same Alpaca-style template used for Stage 2 SFT training -- the "
            "model was trained to expect questions in that format, so DPO must "
            "compare responses under the same format it will actually be asked "
            "at inference time.",
        ]),
        code([
            "from datasets import load_dataset\n",
            "\n",
            f'ALPACA_PROMPT = """{prompt_text}"""\n',
            "\n",
            "def format_dpo_prompt(example):\n",
            '    example["prompt"] = ALPACA_PROMPT.format(example["prompt"], "")\n',
            "    return example\n",
            "\n",
            'dpo_dataset = load_dataset("json", data_files="data/preference_dataset.jsonl", split="train")\n',
            "dpo_dataset = dpo_dataset.map(format_dpo_prompt)\n",
            "dpo_dataset[0]",
        ]),
        markdown([
            "## 3. Configure and run DPO training\n",
            "\n",
            "`beta` controls how strongly the model is pushed toward the chosen "
            "response versus how close it stays to the reference (SFT) model. "
            "`learning_rate` is much lower here than in Stages 1-2 because DPO "
            "only needs to nudge an already-competent model's preferences, not "
            "relearn from scratch. We pass our separately-loaded `ref_model` "
            "explicitly, rather than `ref_model=None`, so the reference is "
            "genuinely the Stage 2 SFT model.",
        ]),
        code([
            "from trl import DPOTrainer, DPOConfig\n",
            "from unsloth import is_bfloat16_supported\n",
            "\n",
            "dpo_trainer = DPOTrainer(\n",
            "    model = model,\n",
            "    ref_model = ref_model,\n",
            "    args = DPOConfig(\n",
            "        max_length = max_seq_length,\n",
            "        per_device_train_batch_size = 2,\n",
            "        gradient_accumulation_steps = 4,\n",
            f"        num_train_epochs = {epochs},\n",
            f"        learning_rate = {learning_rate},\n",
            "        beta = 0.1,\n",
            "        fp16 = not is_bfloat16_supported(),\n",
            "        bf16 = is_bfloat16_supported(),\n",
            "        logging_steps = 5,\n",
            '        output_dir = "stage3_outputs",\n',
            '        optim = "adamw_8bit",\n',
            "        seed = 3407,\n",
            "    ),\n",
            "    train_dataset = dpo_dataset,\n",
            "    processing_class = tokenizer,\n",
            ")\n",
            "dpo_trainer.train()",
        ]),
        markdown(["## 4. Save the DPO-aligned model"]),
        code([
            'model.save_pretrained("stage3_model")\n',
            'tokenizer.save_pretrained("stage3_model")',
        ]),
        markdown(["## 5. Test the model after DPO"]),
        code([
            "import json as _json\n",
            "\n",
            'first_prompt = _json.loads(open("data/preference_dataset.jsonl").readline())["prompt"]\n',
            "\n",
            "FastLanguageModel.for_inference(model)\n",
            'prompt = ALPACA_PROMPT.format(first_prompt, "")\n',
            'inputs = tokenizer([prompt], return_tensors="pt").to("cuda")\n',
            "outputs = model.generate(**inputs, max_new_tokens=128)\n",
            'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        ]),
    ]
    return notebook(cells)
```

- [ ] **Step 4: Run the full notebook template test suite**

```bash
python3 -m pytest tests/test_notebook_templates.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/notebook_templates/stage3.py tests/test_notebook_templates.py
git commit -m "feat: add Stage 3 (DPO) notebook template with explicit ref_model"
```

---

### Task 9: Generator — stage selection logic

**Files:**
- Create: `backend/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generator.py`:

```python
from backend.generator import select_stages


def test_instruction_only():
    assert select_stages(has_raw_text=False, has_preference=False) == ["stage2"]


def test_raw_text_and_instruction():
    assert select_stages(has_raw_text=True, has_preference=False) == ["stage1", "stage2"]


def test_instruction_and_preference():
    assert select_stages(has_raw_text=False, has_preference=True) == ["stage2", "stage3"]


def test_all_three():
    assert select_stages(has_raw_text=True, has_preference=True) == ["stage1", "stage2", "stage3"]
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_generator.py -v
```

Expected: FAIL — `backend/generator.py` does not exist.

- [ ] **Step 3: Write `backend/generator.py`**

```python
"""Ties together validation, stage selection, and notebook generation."""


def select_stages(has_raw_text: bool, has_preference: bool) -> list[str]:
    """Instruction data (Stage 2) is always required and always included.
    Raw text (Stage 1) and preference data (Stage 3) are optional add-ons."""
    stages = []
    if has_raw_text:
        stages.append("stage1")
    stages.append("stage2")
    if has_preference:
        stages.append("stage3")
    return stages
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
python3 -m pytest tests/test_generator.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/generator.py tests/test_generator.py
git commit -m "feat: add pipeline stage selection logic"
```

---

### Task 10: Generator — `generate_project`

**Files:**
- Modify: `backend/generator.py` (append `ValidationFailed`, `generate_project`)
- Modify: `tests/test_generator.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_generator.py -v -k generate_project
```

Expected: FAIL — `generate_project` does not exist.

- [ ] **Step 3: Append to `backend/generator.py`**

```python
import json

from backend.models_config import MODELS
from backend.notebook_templates.stage1 import build_stage1_notebook
from backend.notebook_templates.stage2 import build_stage2_notebook
from backend.notebook_templates.stage3 import build_stage3_notebook
from backend.validators import validate_instruction_jsonl, validate_preference_jsonl, validate_raw_text


class ValidationFailed(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def generate_project(
    domain: str,
    model_key: str,
    instruction_data: str,
    raw_text: str | None = None,
    preference_data: str | None = None,
) -> dict[str, bytes]:
    errors = [f"instruction_dataset.jsonl: {e}" for e in validate_instruction_jsonl(instruction_data)]
    if raw_text is not None:
        errors += [f"non_instruction_data.txt: {e}" for e in validate_raw_text(raw_text)]
    if preference_data is not None:
        errors += [f"preference_dataset.jsonl: {e}" for e in validate_preference_jsonl(preference_data)]
    if model_key not in MODELS:
        errors.append(f"Unknown model '{model_key}'. Choose one of: {', '.join(MODELS)}.")
    if errors:
        raise ValidationFailed(errors)

    model = MODELS[model_key]
    stages = select_stages(has_raw_text=raw_text is not None, has_preference=preference_data is not None)

    files: dict[str, bytes] = {
        "data/instruction_dataset.jsonl": instruction_data.encode("utf-8"),
    }
    if raw_text is not None:
        files["data/non_instruction_data.txt"] = raw_text.encode("utf-8")
    if preference_data is not None:
        files["data/preference_dataset.jsonl"] = preference_data.encode("utf-8")

    previous_adapter = None
    if "stage1" in stages:
        nb1 = build_stage1_notebook(
            domain, model["unsloth_model_id"], model["lora"],
            model["stage1"]["learning_rate"], model["stage1"]["epochs"],
        )
        files["notebooks/non_instruction_finetuning.ipynb"] = json.dumps(nb1, indent=1).encode("utf-8")
        previous_adapter = "stage1_adapter"

    nb2 = build_stage2_notebook(
        domain, model["unsloth_model_id"], model["lora"],
        model["stage2"]["learning_rate"], model["stage2"]["epochs"], previous_adapter,
    )
    files["notebooks/instruction_finetuning.ipynb"] = json.dumps(nb2, indent=1).encode("utf-8")

    if "stage3" in stages:
        nb3 = build_stage3_notebook(
            domain, model["unsloth_model_id"], model["lora"],
            model["stage3"]["learning_rate"], model["stage3"]["epochs"], "stage2_adapter",
        )
        files["notebooks/dpo_alignment.ipynb"] = json.dumps(nb3, indent=1).encode("utf-8")

    files["README.md"] = build_readme(domain, model["display_name"], stages).encode("utf-8")
    return files
```

Note: `build_readme` is referenced here but written in Task 11 — this task will fail to import until Task 11 adds it. Add a temporary minimal stub in this step so Task 10's tests pass on their own, then Task 11 replaces it:

```python
def build_readme(domain: str, model_display_name: str, stages: list[str]) -> str:
    return f"# {domain.title()} Fine-Tuning Project\n"
```

Place this stub above `generate_project` in `backend/generator.py` for now; Task 11 will replace its body with the full version.

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
python3 -m pytest tests/test_generator.py -v
```

Expected: all 8 tests PASS (4 from Task 9 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add backend/generator.py tests/test_generator.py
git commit -m "feat: add generate_project pipeline orchestration"
```

---

### Task 11: Generator — zip building and full README generation

**Files:**
- Modify: `backend/generator.py` (add `build_zip`, replace `build_readme` stub with full version)
- Modify: `tests/test_generator.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_generator.py -v -k "build_zip or build_readme"
```

Expected: FAIL — `build_zip` does not exist, and `build_readme`'s stub doesn't number stages.

- [ ] **Step 3: Replace the `build_readme` stub and add `build_zip` in `backend/generator.py`**

Replace the stub written in Task 10 with:

```python
import io
import zipfile


def build_readme(domain: str, model_display_name: str, stages: list[str]) -> str:
    stage_descriptions = {
        "stage1": "`notebooks/non_instruction_finetuning.ipynb` -- adapts the base model to your raw domain text.",
        "stage2": "`notebooks/instruction_finetuning.ipynb` -- teaches the model to answer questions in your instruction dataset.",
        "stage3": "`notebooks/dpo_alignment.ipynb` -- aligns the model to prefer your chosen responses over rejected ones.",
    }
    steps = "\n".join(f"{i + 1}. {stage_descriptions[s]}" for i, s in enumerate(stages))
    return f"""# {domain.title()} Fine-Tuning Project

Generated by Fine-Tuning Studio for the **{domain}** domain, using
**{model_display_name}** as the base model.

## How to run this on Google Colab

Upload this folder's contents to a Colab session (or clone if you've pushed it
to GitHub), then run each notebook in this order, using a T4 GPU
(Runtime > Change runtime type > T4 GPU):

{steps}

Each notebook saves its output as a folder (e.g. `stage1_adapter`) which the
next notebook expects to find in the same directory -- download it from Colab
and re-upload it to the next notebook's session, or keep the whole pipeline
running in one long-lived Colab session.
"""


def build_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return buffer.getvalue()
```

Move the `import io` and `import zipfile` lines to the top of the file with the other imports rather than leaving them inline.

- [ ] **Step 4: Run the full generator test suite**

```bash
python3 -m pytest tests/test_generator.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/generator.py tests/test_generator.py
git commit -m "feat: add zip packaging and full README generation"
```

---

### Task 12: FastAPI app

**Files:**
- Create: `backend/app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python3 -m pytest tests/test_app.py -v
```

Expected: FAIL — `backend/app.py` does not exist.

- [ ] **Step 3: Write `backend/app.py`**

```python
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
python3 -m pytest tests/test_app.py -v
```

Expected: all 3 tests PASS. Note: `app.mount("/", StaticFiles(directory="frontend", ...))` requires the `frontend/` directory to exist with at least one file, or FastAPI will raise at import time — Task 13 creates `frontend/index.html` before you actually run the server; the `TestClient` import in this task's tests still needs the directory to exist. Create an empty placeholder now if Task 13 hasn't run yet:

```bash
mkdir -p frontend
touch frontend/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add backend/app.py tests/test_app.py frontend/.gitkeep
git commit -m "feat: add FastAPI app with upload/generate endpoints"
```

---

### Task 13: Frontend

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/app.js`
- Create: `frontend/style.css`
- Delete: `frontend/.gitkeep` (no longer needed once real files exist)

- [ ] **Step 1: Write `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Fine-Tuning Studio</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <main>
    <h1>Fine-Tuning Studio</h1>
    <p>Upload your data, pick a domain and a base model, and get back ready-to-run Colab notebooks.</p>

    <form id="generate-form">
      <label>
        Domain name
        <input type="text" id="domain" name="domain" placeholder="e.g. legal, healthcare, customer support" required />
      </label>

      <label>
        Base model
        <select id="model" name="model" required></select>
      </label>

      <label>
        Instruction dataset (.jsonl, required)
        <input type="file" id="instruction_data" name="instruction_data" accept=".jsonl" required />
      </label>

      <label>
        Raw domain text (.txt, optional -- enables Stage 1 non-instruction fine-tuning)
        <input type="file" id="raw_text" name="raw_text" accept=".txt" />
      </label>

      <label>
        Preference dataset (.jsonl, optional -- enables Stage 3 DPO alignment)
        <input type="file" id="preference_data" name="preference_data" accept=".jsonl" />
      </label>

      <button type="submit">Generate</button>
    </form>

    <div id="status"></div>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `frontend/app.js`**

```javascript
async function loadModels() {
  const response = await fetch("/api/models");
  const models = await response.json();
  const select = document.getElementById("model");
  for (const [key, displayName] of Object.entries(models)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = displayName;
    select.appendChild(option);
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const statusEl = document.getElementById("status");
  statusEl.textContent = "Generating...";

  const formData = new FormData(event.target);

  const response = await fetch("/api/generate", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    const errors = Array.isArray(error.detail) ? error.detail : [String(error.detail)];
    statusEl.innerHTML =
      "<strong>Errors:</strong><ul>" + errors.map((e) => `<li>${e}</li>`).join("") + "</ul>";
    return;
  }

  const blob = await response.blob();
  const domain = document.getElementById("domain").value.trim().replace(/\s+/g, "_") || "project";
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${domain}_finetuning.zip`;
  link.click();
  URL.revokeObjectURL(url);
  statusEl.textContent = "Done! Your zip has downloaded.";
}

document.getElementById("generate-form").addEventListener("submit", handleSubmit);
loadModels();
```

- [ ] **Step 3: Write `frontend/style.css`**

```css
body {
  font-family: system-ui, sans-serif;
  max-width: 640px;
  margin: 2rem auto;
  padding: 0 1rem;
  color: #1a1a1a;
}

form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1.5rem;
}

label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-weight: 600;
}

input,
select,
button {
  font-size: 1rem;
  padding: 0.5rem;
}

button {
  cursor: pointer;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 0.75rem;
}

button:hover {
  background: #1d4ed8;
}

#status {
  margin-top: 1.5rem;
}

#status ul {
  color: #b91c1c;
}
```

- [ ] **Step 4: Remove the placeholder and run the backend tests once more**

```bash
rm frontend/.gitkeep
python3 -m pytest tests/test_app.py -v
```

Expected: all 3 tests still PASS (the frontend directory now has real files instead of the placeholder).

- [ ] **Step 5: Manually verify in a browser**

```bash
python3 -m uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000` in a browser. Confirm:
- The model dropdown populates with 4 options.
- Uploading a small valid `.jsonl` instruction file (20+ examples, matching the schema in Task 3) with only that file filled in and clicking Generate downloads a zip.
- Unzip it and confirm it contains `data/instruction_dataset.jsonl`, `notebooks/instruction_finetuning.ipynb`, and `README.md`, and does NOT contain `notebooks/non_instruction_finetuning.ipynb` or `notebooks/dpo_alignment.ipynb`.
- Uploading an invalid file (e.g. fewer than 20 examples) shows the error list on the page instead of downloading a zip.

Stop the server (Ctrl+C) once verified.

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/style.css
git rm frontend/.gitkeep
git commit -m "feat: add frontend upload form"
```

---

### Task 14: Top-level README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Fine-Tuning Studio

A web tool that generates ready-to-run Google Colab notebooks for fine-tuning
a small language model on your own dataset, for any domain.

## What this is (and isn't)

This is a **notebook generator**, not a training service. Upload your dataset,
pick a domain name and a base model, and it validates your data and returns a
zip containing the correct Colab notebooks for your data (1, 2, or 3 stages
depending on what you uploaded), pre-filled with the right model, LoRA config,
and file paths. You still run the notebooks yourself on Colab's free GPU tier
-- nothing here executes training, since there is no GPU available to a web
backend running for free.

This generalizes the pipeline built and debugged in
[finance-faq-assistant-finetuning](https://github.com/ashwiniadik/finance-faq-assistant-finetuning):
the notebook templates carry forward every fix found during that project's
code review (correct TRL API usage, correct `packing` setting per stage,
explicit DPO reference model, consistent prompt templates between SFT and DPO)
rather than re-deriving them from scratch.

## How it works

1. Upload an instruction dataset (`.jsonl`, required) and optionally raw domain
   text (`.txt`) and/or a preference dataset (`.jsonl`).
2. The backend validates each file (schema, minimum size, basic quality checks).
3. Based on which files you provided, it picks 1-3 pipeline stages:
   - instruction only -> SFT-only pipeline
   - + raw text -> adds Stage 1 (non-instruction fine-tuning) before SFT
   - + preference data -> adds Stage 3 (DPO alignment) after SFT
4. It generates the corresponding notebook(s) and a README, zips them with your
   data, and returns the zip.
5. You upload that zip's contents to Colab and run it yourself.

## Supported models

Qwen2.5-0.5B, Qwen2.5-1.5B, TinyLlama-1.1B, Llama-3.2-1B -- all verified to fit
a free Colab T4 GPU with Unsloth + QLoRA.

## Dataset formats

| File | Format | Minimum |
|---|---|---|
| Raw domain text | `.txt`, paragraphs separated by blank lines | 10 paragraphs, 15+ words each |
| Instruction dataset | `.jsonl`, `{"instruction": ..., "response": ...}` per line | 20 examples, 5+ word responses |
| Preference dataset | `.jsonl`, `{"prompt": ..., "chosen": ..., "rejected": ...}` per line | 20 examples |

## Running locally

```bash
pip install -r requirements.txt
python3 -m uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000`.

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

Unlike the Finance project, every part of this system -- validators, notebook
generation, and the API -- is fully testable locally with no GPU required,
since nothing here executes model training.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README"
```

---

### Task 15: End-to-end integration tests across all 4 stage combinations

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_integration.py`:

```python
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
```

- [ ] **Step 2: Run the tests**

```bash
python3 -m pytest tests/test_integration.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 3: Run the entire test suite**

```bash
python3 -m pytest -v
```

Expected: every test across all files passes (validators, model config, notebook templates, generator, app, integration).

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for all 4 stage combinations"
```

---

## Self-Review Notes

- **Spec coverage:** every design section maps to a task — architecture/data flow (Tasks 12-13), model config (Task 5, with IDs verified against Hugging Face rather than assumed), notebook generation with inherited Finance-project fixes (Tasks 6-8), validation including the deliberate preference-length-rule removal (Tasks 2-4), testing at every layer (all tasks are TDD, plus Task 15's cross-combination integration coverage), repository structure (File Structure section, matches exactly).
- **Placeholder scan:** the only bracketed note is Task 10's temporary `build_readme` stub, which is explicitly a real, working piece of code (not a TBD) that Task 11 explicitly replaces in a later step — not a placeholder left unresolved.
- **Type/name consistency:** `alpaca_prompt_text(domain)` is defined once in `_helpers.py` and called identically by both `stage2.py` and `stage3.py`, structurally guaranteeing the two notebooks can't drift apart the way the Finance project's hand-written duplicate definitions could have — checked and enforced by Task 8's `test_stage2_and_stage3_use_identical_alpaca_prompt`. Adapter names (`stage1_adapter` -> `stage2_adapter` -> `stage3_model`) are consistent across every template and the `generate_project` orchestration in Task 10. `select_stages`' return values (`"stage1"`/`"stage2"`/`"stage3"`) are used identically in `generate_project` and `build_readme`.
