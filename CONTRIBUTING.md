# Contributing

Use Python 3.12 and create an isolated virtual environment. Install the CPU PyTorch wheels first,
then the development dependencies:

```bash
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
```

Run `ruff check .`, `ruff format --check .`, and `pytest -q` before opening a pull request. Normal
CI uses controlled ML services and never downloads model weights. Run `pytest -m model` only when
you intentionally want to download and execute the real models.

The common and Heavy route contracts, pinned dependency policy, lazy-loading requirement, and
Python version are shared across the APIZIT reference suite. Contract changes require a
coordinated update of the canonical APIZIT guide and all nine repositories.
