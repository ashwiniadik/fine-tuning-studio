# Contributing

## Setup

```bash
git clone https://github.com/ashwiniadik/fine-tuning-studio.git
cd fine-tuning-studio
pip install -r requirements.txt
```

No GPU, database, or external service is needed to develop or test this
project -- everything runs locally.

## Running the app

```bash
python3 -m uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000`.

## Running the tests

```bash
pytest
```

CI (`.github/workflows/tests.yml`) runs the full suite on every push and pull
request to `main`. A PR won't be considered mergeable unless it passes.

## Project layout

```
backend/
  validators.py              # dataset validation (raw text, instruction, preference, domain)
  models_config.py           # curated base models + LoRA/hyperparameter defaults
  generator.py                # ties validation + stage selection + templates into a downloadable zip
  notebook_templates/         # one module per pipeline stage, built on shared cell-builder helpers
backend/app.py                 # FastAPI routes
frontend/                      # plain HTML/CSS/JS, no build step
tests/                         # one test file per backend module, plus test_integration.py end-to-end
```

`docs/superpowers/specs/` and `docs/superpowers/plans/` hold the original
design spec and implementation plan, if you want the reasoning behind a
decision rather than just the code.

## Conventions

- **Tests first.** Every module here was built test-first; keep doing that
  for new code. `tests/` mirrors `backend/` one-to-one.
- **Validators return errors, they don't raise.** Each `validate_*` function
  in `backend/validators.py` returns a `list[str]` of human-readable error
  messages (empty list = valid). `generator.py` aggregates all of them before
  raising `ValidationFailed`, so a user seeing multiple problems at once gets
  all of them in one round trip, not one-at-a-time.
- **User-controlled strings that reach a downstream interpreter get
  validated, not just escaped.** `domain` is embedded into generated Python
  notebook source (`alpaca_prompt_text`), so it's validated against an
  allowlist both at the API boundary and again inside the notebook-template
  helper itself (defense in depth) -- see `validate_domain`. If you add
  another field that flows into generated code, apply the same pattern rather
  than trusting the caller already checked.
- **Never build HTML from untrusted strings with `innerHTML`.** Error text
  ultimately originates from user-uploaded file content (e.g. JSON keys in a
  bad `.jsonl` upload) and must stay text. `frontend/app.js` renders errors by
  creating DOM nodes and setting `textContent` -- follow that pattern for any
  new UI that displays server-provided error messages.
- **Test what a real browser actually sends, not just what's convenient.**
  `FormData` includes an empty `File` object (`filename=""`) for every
  untouched `<input type="file">` -- httpx's `TestClient` convenience `files=`
  dict doesn't reproduce that wire format. If you're testing upload-handling
  edge cases, check `tests/test_app.py`'s
  `test_generate_with_browser_style_empty_optional_files_returns_zip` for how
  to build a raw multipart body that matches real browser behavior.
- **Floor-pin dependencies in `requirements.txt`.** This app runs and is
  tested locally rather than in a remote-controlled environment, so an
  unpinned install could silently drift to a breaking version between runs.

## Submitting a change

1. Write a failing test.
2. Make it pass.
3. Run the full suite (`pytest`) -- it should stay green.
4. Open a PR. Describe what changed and why; CI will run automatically.
