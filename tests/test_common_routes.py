from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.main import app


def test_exact_route_contract():
    routes = {(route.path, tuple(sorted(route.methods or set()))) for route in app.routes}
    assert routes == {
        ("/health", ("GET",)),
        ("/info", ("GET",)),
        ("/echo", ("POST",)),
        ("/items/{item_id}", ("GET",)),
        ("/slow", ("GET",)),
        ("/ready", ("GET",)),
        ("/text/embedding", ("POST",)),
        ("/text/similarity", ("POST",)),
        ("/image/analyze", ("POST",)),
        ("/image/embedding", ("POST",)),
    }


def test_health_is_lightweight(client, registry):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert not registry.is_loaded("text")
    assert not registry.is_loaded("image")


def test_info_contract(client):
    response = client.get("/info")

    assert response.status_code == 200
    assert response.json()["framework"] == "fastapi"
    assert response.json()["profile"] == "heavy"


def test_echo_round_trip(client):
    response = client.post("/echo", json={"message": "hello", "count": 2})

    assert response.status_code == 200
    assert response.json() == {"received": {"message": "hello", "count": 2}}


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"message": "", "count": 1}, {"message": "hello", "count": True}],
)
def test_echo_rejects_invalid_json(client, payload):
    response = client.post("/echo", json=payload)

    assert response.status_code == 400


def test_item_path_and_query_parameters(client):
    response = client.get("/items/7?include_details=true")

    assert response.status_code == 200
    assert response.json() == {
        "details": "Reference item 7",
        "include_details": True,
        "item_id": 7,
    }


@pytest.mark.parametrize("path", ["/items/0", "/items/7?include_details=maybe"])
def test_item_rejects_invalid_parameters(client, path):
    assert client.get(path).status_code == 400


def test_slow_uses_exact_duration_without_waiting(client, monkeypatch):
    mocked_sleep = AsyncMock()
    monkeypatch.setattr("app.main.asyncio.sleep", mocked_sleep)

    response = client.get("/slow")

    assert response.status_code == 200
    mocked_sleep.assert_awaited_once_with(80)
    assert response.json() == {"delay_seconds": 80, "status": "completed"}


def test_ready_loads_both_models(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "models": {"text": True, "image": True},
        "device": "cpu",
        "version": "1.0.0",
    }
