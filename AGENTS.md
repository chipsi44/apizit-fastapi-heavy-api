# Repository Instructions

This repository is the public Heavy FastAPI reference API for APIZIT. Keep it standalone,
deployable with Python 3.12, and aligned with the canonical conventions in the APIZIT platform
repository at `docs/reference-api-repository-conventions.md`.

The ten-route contract is deliberate. Do not add, remove, or rename public routes without a
coordinated update to the canonical guide and every reference repository. `/health` must stay
immediate and must never load models. `/slow` must call an 80-second wait but must not be used as
a health check.

Keep all direct dependencies pinned. Retain the real CPU ML stack and lazy model loading. Normal
tests must stub model services and must not download weights; `pytest -m model` is the explicit
real-model check. Never add credentials, model weights, caches, a Dockerfile, or APIZIT internals.

Before committing, run `ruff check .`, `ruff format --check .`, `pytest -q`, and the import smoke
test documented in the README.
