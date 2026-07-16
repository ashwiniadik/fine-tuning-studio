# Fine-Tuning Studio — Design

## Purpose

A reusable web tool that generalizes the Unsloth fine-tuning workflow built for the
Finance FAQ Assistant (`finance-faq-assistant-finetuning`) into something that works
for any domain, any dataset, and any of a curated set of small models. Built as a
portfolio/learning project: it demonstrates the fine-tuning pipeline as a reusable
system rather than a one-off script, which is a stronger interview story than a
single fine-tuned model.

**Hard constraint:** no GPU is available anywhere in this environment, and Unsloth
requires CUDA (it does not support CPU or Apple Silicon/MPS training). This rules out
any design where the backend itself executes training. The tool therefore automates
everything *around* training — validating data, generating ready-to-run notebooks,
picking hyperparameters — while actual training still happens on the user's own free
Google Colab GPU, exactly as it did for the Finance project's Task 12.

## Architecture & Data Flow

A single Python process (FastAPI) serves both a small REST API and a static
one-page frontend (plain HTML/CSS/JS, no build step, no npm).

**Flow:**
1. User opens the page, enters a domain name (free text, used for naming and prompt
   wording), picks a base model from a curated dropdown, and uploads 1-3 files:
   raw text (`.txt`), instruction pairs (`.jsonl`), and/or preference triples
   (`.jsonl`). Instruction data is **required** (it's the only stage that actually
   teaches Q&A behavior); raw text and preference data are optional.
2. Backend **validates** each uploaded file (schema/quality checks, see below).
   Invalid uploads return clear errors before anything is generated.
3. Backend **determines which pipeline stages apply**, based on which files were
   provided. Four valid combinations:
   - instruction only → SFT-only pipeline (one notebook, trains directly from the
     base model)
   - raw text + instruction → Stage 1 (non-instruction FT) → Stage 2 (SFT)
   - instruction + preference → Stage 2 (SFT) → Stage 3 (DPO)
   - raw text + instruction + preference → full 3-stage pipeline (Stage 1 → 2 → 3),
     matching the Finance project exactly
4. Backend **generates** the corresponding notebook(s) from parameterized templates,
   filling in the domain name, chosen model's config, correct dataset filenames, and
   correct adapter/model folder names so multi-notebook pipelines chain correctly.
5. Backend **packages** the validated data files, generated notebook(s), and a short
   generated README (explaining what was generated and how to run it on Colab) into
   a zip and returns it for download.

No training ever runs on the backend. No database, no job queue, no user accounts —
the whole thing is a stateless request → zip response. The user takes the zip to
Colab and runs it themselves, the same manual workflow as the Finance project's
Task 12.

## Model Configuration

Curated list (from the original assignment's recommendations, all verified to fit
a free Colab T4 with Unsloth + QLoRA):

- Qwen2.5-0.5B
- Qwen2.5-1.5B
- TinyLlama-1.1B
- Llama-3.2-1B

Each model has a config entry: Unsloth model ID, LoRA `r`/`alpha`/`dropout`/
`target_modules`, and default learning rates/epochs/batch size per stage. Defaults
reuse the exact values validated in the Finance project (`r=16, alpha=16,
dropout=0`, the same 7-module attention+MLP `target_modules` list). This target
module list is expected to hold across all four models since they share the
standard Llama-family attention/MLP module naming — this should be spot-checked
once notebooks are actually generated for each model, not assumed blindly.

## Notebook Generation

Notebook templates are the Finance project's three notebooks
(`non_instruction_finetuning.ipynb`, `instruction_finetuning.ipynb`,
`dpo_alignment.ipynb`), generalized into parameterized templates (domain name,
model ID, LoRA config, dataset filenames, adapter names become placeholders).

Critically, these templates **inherit every fix found during the Finance project's
code reviews**, rather than re-deriving them from scratch:
- `processing_class = tokenizer` (not the deprecated `tokenizer=` kwarg)
- Correct `packing=True` for Stage 1 (continuous raw text) vs. `packing=False` for
  Stage 2 (discrete instruction examples)
- An explicit, separately-loaded `ref_model` for DPO (not `ref_model=None`, which
  was found to give the wrong reference model when resuming an already-adapted
  checkpoint under the TRL version Unsloth pins)
- `is_bfloat16_supported()`-based fp16/bf16 selection
- `data/`-prefixed dataset paths matching the generated zip's actual layout
- The DPO preference dataset's `prompt` field wrapped in the same Alpaca-style
  template used for SFT training (avoiding the train/serve prompt-format mismatch
  found in the Finance project's DPO notebook)

## Data Validation

Generalized versions of the Finance project's dataset checks (`tests/test_datasets.py`),
loosened from that project's assignment-specific minimums to sensible generic ones:

| File | Schema | Minimum count | Quality checks |
|---|---|---|---|
| Raw text (`.txt`) | paragraphs separated by blank lines | 10 | each paragraph >= 15 words |
| Instruction (`.jsonl`) | `{instruction, response}`, exactly these keys | 20 | non-empty instruction, response >= 5 words |
| Preference (`.jsonl`) | `{prompt, chosen, rejected}`, exactly these keys | 20 | both non-empty, `chosen != rejected` |

**One deliberate change from the Finance project**: the preference validator does
**not** require `chosen` to be longer than `rejected`. The Finance project's tests
enforced that inequality, but code review during that project flagged it as a
length-confound anti-pattern — it risks teaching DPO to prefer verbosity over
correctness, since a model can satisfy "chosen is always longer" without learning
anything about actual quality. The generalized validator only requires
`chosen != rejected` and both fields non-empty.

## Testing

Unlike the Finance project's notebooks, all of this backend logic is genuinely
testable locally, with no GPU required anywhere:

- `pytest` tests for the validation logic (schema/count/quality checks per file
  type, parameterized similarly to the Finance project's `tests/test_datasets.py`)
- `pytest` tests for notebook generation: generated JSON is valid, the correct
  stage combination is selected for each of the 4 input combinations, parameters
  are correctly substituted, and adapter/model names chain consistently across
  multi-notebook pipelines
- FastAPI `TestClient` tests for the API endpoints (upload -> validate -> generate
  -> zip download), which are fully runnable end-to-end here, unlike the Finance
  project's Colab-only notebooks
- Frontend: manual browser verification (plain HTML/JS, no framework, so no
  automated frontend test suite) — driven in an actual browser before considering
  the UI done, not just assumed to work from reading the code

## Repository Structure

New, separate repository (`fine-tuning-studio`), independent from
`finance-faq-assistant-finetuning`:

```
backend/
  app.py               FastAPI app: routes for upload/validate/generate
  validators.py        generalized dataset validation logic
  models_config.py     curated model list + per-model LoRA/hyperparameter defaults
  notebook_templates/  parameterized notebook templates (ported from finance-faq-assistant-finetuning)
  generator.py         selects pipeline stages, fills templates, builds the zip
frontend/
  index.html           single-page upload form (domain, model dropdown, file inputs)
  app.js               form handling, fetch calls to the API
  style.css
tests/
  test_validators.py
  test_generator.py
  test_api.py
README.md
requirements.txt
```

## Out of Scope

Explicitly excluded to keep this buildable without cost or unbounded complexity:

- No actual training execution or orchestration — the backend never touches a GPU
- No support for arbitrary/unlisted Hugging Face models, only the curated list
- No LLM-assisted generation of training data from raw/unstructured uploads (would
  require paid LLM API calls)
- No automated GitHub repo creation/push — output is a zip download only
- No user accounts, job history, or persistence between requests
