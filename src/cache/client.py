"""Redis connection helpers, cache key layout, and invalidation."""

from __future__ import annotations

import json
import os
from typing import Any

import redis
from redis.asyncio import Redis

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
CACHE_TTL_SECONDS = 300
CACHE_PATTERNS = ("rec:*", "sim:*")


def redis_url() -> str:
    return os.environ.get("REDIS_URL") or DEFAULT_REDIS_URL


def rec_key(user_id: int, k: int) -> str:
    return f"rec:{user_id}:{k}"


def sim_key(product_id: str, k: int) -> str:
    return f"sim:{product_id}:{k}"


async def get_cached_json(client: Redis, key: str) -> Any | None:
    raw = await client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def set_cached_json(
    client: Redis,
    key: str,
    payload: Any,
    ttl: int = CACHE_TTL_SECONDS,
) -> None:
    await client.set(key, json.dumps(payload), ex=ttl)


async def invalidate_recommendation_cache(client: Redis) -> int:
    """Delete recommendation and similar-item cache keys (async)."""
    deleted = 0
    for pattern in CACHE_PATTERNS:
        async for key in client.scan_iter(match=pattern):
            deleted += int(await client.delete(key))
    return deleted


def invalidate_recommendation_cache_sync(client: redis.Redis | None = None) -> int:
    """Delete recommendation and similar-item cache keys (sync).

    Used by batch jobs such as metadata reloads that do not run an event loop.
    """
    close = False
    if client is None:
        client = redis.Redis.from_url(redis_url(), decode_responses=True)
        close = True
    try:
        deleted = 0
        for pattern in CACHE_PATTERNS:
            for key in client.scan_iter(match=pattern):
                deleted += int(client.delete(key))
        return deleted
    finally:
        if close:
            client.close()
