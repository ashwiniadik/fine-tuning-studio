## Summary

<!-- What changed, and why. Link an issue if there is one. -->

## Test plan

<!-- How you verified this. `pytest` passing is necessary but often not
sufficient -- if you touched backend/app.py or frontend/, confirm you drove
the actual running app (see CONTRIBUTING.md), not just the test suite. -->

- [ ] `pytest` passes locally
- [ ] Ran the app and exercised the affected path directly (not just unit tests)
- [ ] Added/updated tests covering this change

## Checklist

- [ ] No new `innerHTML`/unescaped user input reaching the frontend or a generated notebook (see CONTRIBUTING.md conventions)
- [ ] `CHANGELOG.md` updated under `[Unreleased]` if this is a user-facing change
