"""Shared fixtures for API, database, and recommender tests."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix


class FakeALSModel:
    """Deterministic stand-in for implicit ALS used by API tests."""

    def __init__(self, n_items: int = 2):
        self.n_items = n_items

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        n = min(int(N), self.n_items)
        item_ids = np.arange(n, dtype=np.int32)
        scores = np.linspace(0.9, 0.2, n, dtype=np.float32)
        return item_ids, scores

    def similar_items(self, itemid, N=10):
        n = min(int(N), self.n_items)
        order = [int(itemid)] + [i for i in range(self.n_items) if i != int(itemid)]
        order = order[:n]
        scores = np.linspace(1.0, 0.25, len(order), dtype=np.float32)
        return np.asarray(order, dtype=np.int32), scores


MAIN_GLOBALS = (
    "model",
    "user_item_matrix",
    "user_lookup",
    "product_lookup",
    "user_id_to_idx",
    "product_id_to_idx",
)


@pytest.fixture(autouse=True)
def restore_api_globals():
    """Keep src.api.main artifact state isolated between tests."""
    import src.api.main as main

    snapshot = {name: getattr(main, name) for name in MAIN_GLOBALS}
    yield
    for name, value in snapshot.items():
        setattr(main, name, value)


def install_test_artifacts(model: FakeALSModel | None = None) -> None:
    import src.api.main as main

    main.model = model or FakeALSModel()
    main.user_item_matrix = csr_matrix(np.eye(2, dtype=np.float32))
    main.user_lookup = np.array([100, 200])
    main.product_lookup = np.array(["B001", "B002"])
    main.user_id_to_idx = {100: 0, 200: 1}
    main.product_id_to_idx = {"B001": 0, "B002": 1}


@pytest.fixture
def catalog_row() -> dict:
    return {
        "asin": "B002",
        "title": "Portal 2",
        "price": 9.99,
        "im_url": "http://example.com/portal2.jpg",
        "brand": "Valve",
    }


@pytest.fixture
def mock_db_pool(catalog_row):
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[catalog_row])
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_redis():
    client = MagicMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def api_client(mock_db_pool, mock_redis):
    """FastAPI TestClient with lifespan patched away from Postgres/Redis/artifacts."""

    def _load_artifacts() -> None:
        install_test_artifacts()

    with (
        patch("src.api.main.load_artifacts", side_effect=_load_artifacts),
        patch("src.api.main.asyncpg.create_pool", new=AsyncMock(return_value=mock_db_pool)),
        patch("src.api.main.Redis.from_url", return_value=mock_redis),
    ):
        from src.api.main import app

        with TestClient(app) as client:
            yield client
