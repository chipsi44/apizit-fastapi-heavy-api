"""Domain errors returned without exposing implementation details."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class APIError(Exception):
    code: str
    message: str
    status_code: int


class ModelUnavailableError(APIError):
    def __init__(self, message: str) -> None:
        super().__init__("MODEL_UNAVAILABLE", message, 503)


def error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}
