# APIZIT FastAPI Heavy API

A standalone FastAPI reference project for testing larger APIZIT builds and CPU machine-learning
workloads. It exposes the five common reference routes plus text and image inference routes. Model
services are loaded lazily, so `/health` remains immediate and never downloads weights.

## Run locally

Python 3.12 is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The first call to `/ready` or an ML route downloads the configured pretrained models. `/health`
does not. The `/slow` route intentionally waits exactly 80 seconds and is only a timeout test.

## Routes

- `GET /health`
- `GET /info`
- `POST /echo`
- `GET /items/{item_id}?include_details=true`
- `GET /slow`
- `GET /ready`
- `POST /text/embedding`
- `POST /text/similarity`
- `POST /image/analyze`
- `POST /image/embedding`

Image endpoints expect a multipart upload named `file`. JSON examples:

```json
{"message": "hello", "count": 2}
```

```json
{"text": "Deploy a Python API"}
```

## Verify

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
python -c "from app.main import app; assert app is not None"
```

Normal tests use controlled model services and download no weights. Run `pytest -m model` to opt
into a real-model execution. This project intentionally has no Dockerfile: APIZIT owns the build.
