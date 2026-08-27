"""FastAPI dependencies for the recommendation API."""

import asyncpg
from fastapi import Request


async def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the asyncpg pool attached to FastAPI app state."""
    return request.app.state.pool
