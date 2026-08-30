from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from app.ml import ModelRegistry


def test_registry_reuses_one_instance_across_threads():
    registry = ModelRegistry()
    instance = object()
    calls = 0
    calls_lock = Lock()

    def factory():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.01)
        return instance

    registry.register("model", factory)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: registry.get("model"), range(16)))

    assert calls == 1
    assert all(result is instance for result in results)
