"""Explicit FastAPI application and the complete Heavy reference route contract."""

from __future__ import annotations

import asyncio
import logging
import platform
from importlib.metadata import version
from typing import Annotated

import cv2
import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
import transformers
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictInt
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.errors import APIError, ModelUnavailableError, error_payload
from app.ml import ImageService, ModelRegistry, TextService, decode_image

logger = logging.getLogger(__name__)
SLOW_RESPONSE_SECONDS = 80

app = FastAPI(
    title="APIZIT FastAPI Heavy API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.state.settings = Settings()
app.state.model_registry = ModelRegistry()
app.state.model_registry.register("text", lambda: TextService(app.state.settings))
app.state.model_registry.register("image", lambda: ImageService(app.state.settings))


class EchoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    count: StrictInt


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class SimilarityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: str
    right: str


def _registry() -> ModelRegistry:
    return app.state.model_registry


def _text_field(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise APIError("INVALID_REQUEST", f"The field '{field}' must not be empty.", 400)
    max_length = app.state.settings.max_text_length
    if len(value) > max_length:
        raise APIError(
            "TEXT_TOO_LONG",
            f"The field '{field}' exceeds the {max_length} character limit.",
            400,
        )
    return value


@app.exception_handler(APIError)
async def handle_api_error(_request: Request, error: APIError) -> JSONResponse:
    return JSONResponse(error_payload(error.code, error.message), status_code=error.status_code)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        error_payload("INVALID_REQUEST", "The request does not match the expected contract."),
        status_code=400,
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        error_payload("HTTP_ERROR", str(error.detail)), status_code=error.status_code
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_request: Request, error: Exception) -> JSONResponse:
    logger.exception("Unhandled request error", exc_info=error)
    return JSONResponse(error_payload("INTERNAL_ERROR", "An unexpected error occurred."), 500)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
async def info() -> dict[str, object]:
    settings = app.state.settings
    return {
        "version": settings.version,
        "framework": "fastapi",
        "profile": "heavy",
        "python": platform.python_version(),
        "libraries": {
            "fastapi": version("fastapi"),
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "sentence_transformers": version("sentence-transformers"),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
        "models": {"text": settings.text_model_id, "image": settings.image_model_id},
        "device": "cpu",
    }


@app.post("/echo")
async def echo(payload: EchoRequest) -> dict[str, object]:
    message = _text_field(payload.message, "message")
    return {"received": {"message": message, "count": payload.count}}


@app.get("/items/{item_id}")
async def item(
    item_id: int, include_details: Annotated[bool, Query()] = False
) -> dict[str, object]:
    if item_id < 1:
        raise APIError("INVALID_REQUEST", "The item ID must be a positive integer.", 400)
    response: dict[str, object] = {
        "item_id": item_id,
        "include_details": include_details,
    }
    if include_details:
        response["details"] = f"Reference item {item_id}"
    return response


@app.get("/slow")
async def slow() -> dict[str, object]:
    await asyncio.sleep(SLOW_RESPONSE_SECONDS)
    return {"delay_seconds": SLOW_RESPONSE_SECONDS, "status": "completed"}


@app.get("/ready")
async def ready() -> JSONResponse:
    model_status: dict[str, bool] = {}
    failures: list[str] = []
    for name in ("text", "image"):
        try:
            _registry().get(name)
            model_status[name] = True
        except ModelUnavailableError as error:
            model_status[name] = False
            failures.append(error.message)
    payload: dict[str, object] = {
        "status": "ready" if not failures else "unavailable",
        "models": model_status,
        "device": "cpu",
        "version": app.state.settings.version,
    }
    if failures:
        payload["error"] = {"code": "MODEL_UNAVAILABLE", "message": " ".join(failures)}
        return JSONResponse(payload, status_code=503)
    return JSONResponse(payload)


@app.post("/text/embedding")
async def text_embedding(payload: TextRequest) -> dict[str, object]:
    text = _text_field(payload.text, "text")
    service: TextService = _registry().get("text")
    vector = service.embed([text])[0]
    return {
        "model": service.model_id,
        "dimension": int(vector.shape[0]),
        "embedding": vector.tolist(),
    }


@app.post("/text/similarity")
async def text_similarity(payload: SimilarityRequest) -> dict[str, object]:
    left = _text_field(payload.left, "left")
    right = _text_field(payload.right, "right")
    service: TextService = _registry().get("text")
    return {"model": service.model_id, "similarity": service.similarity(left, right)}


async def _uploaded_image(upload: UploadFile) -> object:
    settings = app.state.settings
    data = await upload.read(settings.max_upload_bytes + 1)
    return decode_image(data, upload.content_type or "", settings)


@app.post("/image/analyze")
async def image_analyze(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    decoded = await _uploaded_image(file)
    service: ImageService = _registry().get("image")
    return {
        "model": service.model_id,
        "image": {"width": decoded.width, "height": decoded.height, "format": decoded.format},
        "predictions": service.classify(decoded),
    }


@app.post("/image/embedding")
async def image_embedding(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    decoded = await _uploaded_image(file)
    service: ImageService = _registry().get("image")
    vector = service.embed(decoded)
    return {
        "model": service.model_id,
        "dimension": int(vector.shape[0]),
        "embedding": vector.tolist(),
    }
