# Security Policy

## Supported Versions

This is a small, actively-developed project with a single line of
development. Only the latest commit on `main` is supported; there are no
maintained release branches.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately by emailing **ashwiniadik23@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce it (a minimal example is ideal)
- Any suggested fix, if you have one

You should get an acknowledgment within a few days. Once a fix is available,
it will be released on `main` and noted in `CHANGELOG.md`; you're welcome to
ask for credit in the writeup or to remain anonymous.

## Scope

This app is a stateless notebook generator: it validates an uploaded
dataset, fills notebook templates, and returns a zip. It does not run model
training, hold a database, or store uploaded data beyond the request that
processes it. Reports most relevant to this project's actual attack surface:

- Input handling in `backend/validators.py` and `backend/app.py` (file
  uploads, form fields such as `domain`, error messages that reach the
  frontend)
- Anything that lets user-controlled input reach a downstream interpreter
  unsanitized -- e.g. the generated notebook's Python source, or the
  frontend's rendered HTML
- Dependency vulnerabilities in `requirements.txt`

Out of scope: vulnerabilities in Google Colab itself, or in the model
weights/training code the generated notebooks download from Hugging Face --
those aren't part of this project.
