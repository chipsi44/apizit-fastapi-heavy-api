"""Lazy CPU model registry and multimodal inference services."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from app.config import Settings
from app.errors import APIError, ModelUnavailableError

logger = logging.getLogger(__name__)
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class ModelRegistry:
    """Load each expensive model once and reuse it for subsequent requests."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._instances: dict[str, Any] = {}
        self._lock = RLock()

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        with self._lock:
            if name in self._instances:
                raise RuntimeError(f"Cannot replace loaded model service: {name}")
            self._factories[name] = factory

    def get(self, name: str) -> Any:
        instance = self._instances.get(name)
        if instance is not None:
            return instance
        with self._lock:
            instance = self._instances.get(name)
            if instance is not None:
                return instance
            factory = self._factories.get(name)
            if factory is None:
                raise ModelUnavailableError(f"The '{name}' model service is not configured.")
            try:
                logger.info("Loading %s model service", name)
                instance = factory()
            except ModelUnavailableError:
                raise
            except Exception as error:
                logger.exception("Failed to load %s model service", name)
                raise ModelUnavailableError(
                    f"The '{name}' model could not be loaded. Check the server logs."
                ) from error
            self._instances[name] = instance
            return instance

    def is_loaded(self, name: str) -> bool:
        return name in self._instances


class TextService:
    """Sentence Transformer embeddings and semantic similarity on CPU."""

    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.text_model_id
        cache_dir = Path(settings.model_cache_dir) / "huggingface"
        self._model = SentenceTransformer(
            self.model_id,
            cache_folder=str(cache_dir),
            device="cpu",
        ).eval()
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            embeddings = self._model.encode(
                texts,
                batch_size=min(len(texts), 8),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def similarity(self, left: str, right: str) -> float:
        embeddings = self.embed([left, right])
        return float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0])


@dataclass(frozen=True, slots=True)
class DecodedImage:
    image: Image.Image
    width: int
    height: int
    format: str


def decode_image(data: bytes, mime_type: str, settings: Settings) -> DecodedImage:
    """Validate an image and enhance its lightness channel with OpenCV."""

    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise APIError("UNSUPPORTED_MEDIA_TYPE", "Only JPEG, PNG, and WebP are supported.", 415)
    if not data:
        raise APIError("INVALID_IMAGE", "The uploaded image is empty.", 400)
    if len(data) > settings.max_upload_bytes:
        raise APIError(
            "PAYLOAD_TOO_LARGE",
            f"The image exceeds the {settings.max_upload_bytes} byte limit.",
            413,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as candidate:
                image_format = candidate.format or ""
                width, height = candidate.size
                if width * height > settings.max_image_pixels:
                    raise APIError(
                        "IMAGE_TOO_LARGE",
                        f"The image exceeds the {settings.max_image_pixels} pixel limit.",
                        413,
                    )
                candidate.verify()
            with Image.open(BytesIO(data)) as candidate:
                image = ImageOps.exif_transpose(candidate).convert("RGB")
    except APIError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise APIError("IMAGE_TOO_LARGE", "The image dimensions are unsafe.", 413) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise APIError("INVALID_IMAGE", "The file is not a valid supported image.", 400) from error
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise APIError(
            "UNSUPPORTED_MEDIA_TYPE",
            "The encoded image must be JPEG, PNG, or WebP.",
            415,
        )
    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    enhanced_lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, green_red, blue_yellow))
    enhanced_image = Image.fromarray(cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB))
    return DecodedImage(enhanced_image, image.width, image.height, image_format)


class ImageService:
    """Pretrained ResNet50 classification and avgpool embeddings on CPU."""

    def __init__(self, settings: Settings) -> None:
        self.model_id = "resnet50"
        weights = ResNet50_Weights.IMAGENET1K_V2
        torch.hub.set_dir(str(Path(settings.model_cache_dir) / "torch"))
        model = resnet50(weights=weights).eval()
        self._model = model
        self._feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval()
        self._transform = weights.transforms()
        self._categories: list[str] = list(weights.meta["categories"])
        self.dimension = 2048

    def _tensor(self, decoded: DecodedImage) -> torch.Tensor:
        return self._transform(decoded.image).unsqueeze(0)

    def classify(self, decoded: DecodedImage, top_k: int = 5) -> list[dict[str, float | str]]:
        with torch.inference_mode():
            logits = self._model(self._tensor(decoded))[0].cpu().numpy()
        probabilities = softmax(logits)
        indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            {"label": self._categories[int(index)], "score": float(probabilities[index])}
            for index in indices
        ]

    def embed(self, decoded: DecodedImage) -> np.ndarray:
        with torch.inference_mode():
            vector = self._feature_extractor(self._tensor(decoded)).flatten(1)[0]
        return vector.cpu().numpy().astype(np.float32, copy=False)
