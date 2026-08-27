"""Integration tests for FastAPI recommendation and similarity endpoints."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from src.api.deps import get_db_pool, get_redis
from src.api.main import hydrate_recommendations
from src.api.schemas import ProductRecommendation


def test_recommend_by_matrix_index_returns_hydrated_products(api_client, mock_db_pool):
    response = api_client.get("/recommend/0?k=2")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["product_id"] == "B001"
    assert payload[1]["product_id"] == "B002"
    assert payload[1]["title"] == "Portal 2"
    assert payload[1]["brand"] == "Valve"
    assert payload[1]["price"] == 9.99
    assert payload[0]["title"] is None
    mock_db_pool.fetch.assert_awaited()


def test_recommend_by_raw_user_id(api_client):
    response = api_client.get("/recommend/100?k=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["product_id"] == "B001"
    assert "score" in payload[0]


def test_recommend_unknown_user_returns_404(api_client):
    response = api_client.get("/recommend/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "user_id 999 not found"


def test_recommend_invalid_user_id_returns_422(api_client):
    response = api_client.get("/recommend/not-an-int")

    assert response.status_code == 422


def test_recommend_cache_hit_skips_database(api_client, mock_redis, mock_db_pool):
    cached = [
        {
            "product_id": "CACHED",
            "score": 0.42,
            "title": "From cache",
            "price": 1.0,
            "im_url": None,
            "brand": None,
        }
    ]
    mock_redis.get = AsyncMock(return_value=json.dumps(cached))

    response = api_client.get("/recommend/0?k=10")

    assert response.status_code == 200
    assert response.json() == cached
    mock_db_pool.fetch.assert_not_called()
    mock_redis.set.assert_not_called()


def test_similar_returns_neighbors_excluding_query_item(api_client):
    response = api_client.get("/similar/B001?k=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["product_id"] == "B002"
    assert payload[0]["title"] == "Portal 2"


def test_similar_cache_hit_skips_database(api_client, mock_redis, mock_db_pool):
    cached = [
        {
            "product_id": "SIMCACHED",
            "score": 0.77,
            "title": "Cached neighbor",
            "price": None,
            "im_url": None,
            "brand": None,
        }
    ]
    mock_redis.get = AsyncMock(return_value=json.dumps(cached))

    response = api_client.get("/similar/B001?k=10")

    assert response.status_code == 200
    assert response.json() == cached
    mock_db_pool.fetch.assert_not_called()


def test_similar_unknown_product_returns_404(api_client):
    response = api_client.get("/similar/MISSING")

    assert response.status_code == 404
    assert response.json()["detail"] == "product_id MISSING not found"


def test_hydrate_recommendations_maps_catalog_and_missing_rows():
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "asin": "B002",
                "title": "Portal 2",
                "price": 9.99,
                "im_url": "http://img",
                "brand": "Valve",
            }
        ]
    )
    items = [("B001", 0.5), ("B002", 0.9)]

    hydrated = asyncio.run(hydrate_recommendations(pool, items))

    assert [item.product_id for item in hydrated] == ["B001", "B002"]
    assert hydrated[0].title is None
    assert hydrated[1] == ProductRecommendation(
        product_id="B002",
        score=0.9,
        title="Portal 2",
        price=9.99,
        im_url="http://img",
        brand="Valve",
    )


def test_dependency_helpers_read_app_state():
    class State:
        pool = object()
        redis = object()

    class App:
        state = State()

    class Request:
        app = App()

    request = Request()
    assert asyncio.run(get_db_pool(request)) is State.pool
    assert asyncio.run(get_redis(request)) is State.redis
