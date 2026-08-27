"""FastAPI dependencies for the recommendation API."""

import asyncpg
from fastapi import Request
from redis.asyncio import Redis


async def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the asyncpg pool attached to FastAPI app state."""
    return request.app.state.pool


async def get_redis(request: Request) -> Redis:
    """Return the Redis client attached to FastAPI app state."""
    return request.app.state.redis
