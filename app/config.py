"""Environment-driven configuration for the standalone FastAPI application."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "apizit-fastapi-heavy-api"
    version: str = "1.0.0"
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "models")
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_id: str = "resnet50-imagenet1k-v2"
