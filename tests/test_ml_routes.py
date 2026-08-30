from __future__ import annotations

import numpy as np
import pytest

from app.ml import TextService


def test_text_embedding_contract(client):
    response = client.post("/text/embedding", json={"text": "Deploy with APIZIT"})

    assert response.status_code == 200
    assert response.json() == {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 3,
        "embedding": [0.25, 0.5, 0.75],
    }


def test_text_similarity_contract(client):
    response = client.post(
        "/text/similarity",
        json={"left": "Deploy a FastAPI API", "right": "Publish a Python service"},
    )

    assert response.status_code == 200
    assert response.json()["similarity"] == 0.8125


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": 123}, {"text": "   "}, {"text": "x" * 33}],
)
def test_text_embedding_validation(client, payload):
    assert client.post("/text/embedding", json=payload).status_code == 400


def test_similarity_uses_cosine_similarity(monkeypatch):
    service = object.__new__(TextService)
    embeddings = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(service, "embed", lambda _texts: embeddings)

    assert service.similarity("left", "right") == pytest.approx(2**-0.5)


def test_image_analyze_contract(client, png_bytes):
    response = client.post(
        "/image/analyze",
        files={"file": ("sample.png", png_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["image"] == {"width": 24, "height": 16, "format": "PNG"}
    assert body["predictions"][0] == {"label": "orange", "score": 0.8}


def test_image_embedding_contract(client, png_bytes):
    response = client.post(
        "/image/embedding",
        files={"file": ("sample.png", png_bytes, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == 2048
    assert body["embedding"][:3] == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("content", "mime_type", "status", "code"),
    [
        (b"not-an-image", "text/plain", 415, "UNSUPPORTED_MEDIA_TYPE"),
        (b"not-an-image", "image/png", 400, "INVALID_IMAGE"),
    ],
)
def test_image_rejects_invalid_files(client, content, mime_type, status, code):
    response = client.post(
        "/image/analyze",
        files={"file": ("sample", content, mime_type)},
    )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
