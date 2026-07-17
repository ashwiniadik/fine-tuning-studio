# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Replaced `httpx` with `httpx2` in `requirements.txt` to resolve a
  `StarletteDeprecationWarning` from `TestClient` on newer Starlette.

### Changed

- Bumped `actions/checkout` and `actions/setup-python` to their latest major
  versions (v7 / v6) so CI runs on the Node 24 runtime without a deprecation
  warning.

### Added

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- `LICENSE` (MIT).
- CI: a GitHub Actions workflow (`.github/workflows/tests.yml`) running the
  test suite on every push/PR to `main`.
- README: status badges (tests, license), an app screenshot, and a repo
  social preview image.

## [1.0.0] - 2026-07-17

Initial release. A FastAPI + plain-HTML tool that validates an uploaded
dataset and generates a zip of ready-to-run Google Colab notebooks for
fine-tuning a curated small language model, for any domain.

### Added

- Dataset validators: raw text, instruction JSONL, preference JSONL, and
  domain name (the last added after a live-browser test showed an
  unrestricted domain could break the Python syntax of a generated notebook,
  or crash it at `.format()` time in Colab).
- A curated model registry (Qwen2.5-0.5B, Qwen2.5-1.5B, TinyLlama-1.1B,
  Llama-3.2-1B) with LoRA and per-stage hyperparameter defaults.
- Notebook templates for all three pipeline stages: non-instruction
  fine-tuning, instruction fine-tuning (SFT), and DPO preference alignment
  with an explicit, correctly-loaded reference model.
- Pipeline orchestration (`generate_project`): stage selection based on which
  optional files were uploaded, notebook generation, README generation, and
  zip packaging.
- A FastAPI backend (`/api/models`, `/api/generate`) and a plain HTML/CSS/JS
  frontend upload form, with no build step.
- An end-to-end integration test suite covering all four valid upload
  combinations and every curated model.

### Fixed

Issues found and fixed via code review and live testing during development,
before the initial tag:

- A `Content-Disposition` header-injection bug and an unhandled non-UTF-8
  upload crash in the upload endpoint.
- A path bug where the frontend's static file mount resolved relative to the
  process's working directory instead of the module's location.
- A reflected-XSS vulnerability in the frontend's error rendering (raw error
  text, which can originate from an uploaded file's JSON keys, was
  concatenated into `innerHTML`).
- A backend crash when a preference dataset's `chosen`/`rejected` fields were
  non-string values.
- A bug where real browsers' `FormData` sends an empty `File` object for
  every untouched optional file input, which the backend treated as
  "provided" rather than "absent" -- breaking every real-browser submission
  that didn't fill in every optional upload. Found by driving the running
  app with an actual browser rather than relying on `curl`/`TestClient`
  alone.

[Unreleased]: https://github.com/ashwiniadik/fine-tuning-studio/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ashwiniadik/fine-tuning-studio/releases/tag/v1.0.0
