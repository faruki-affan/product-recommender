"""FastAPI service that serves ALS product recommendations from saved artifacts."""

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import implicit  # noqa: F401
import numpy as np
import scipy.sparse
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from implicit.cpu.als import AlternatingLeastSquares

from src.api.deps import get_db_pool
from src.api.schemas import ProductRecommendation

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_DATABASE_URL = "postgresql://postgres:9710@localhost:5432/recommender_db"
METADATA_QUERY = (
    "SELECT asin, title, price, im_url, brand FROM products WHERE asin = ANY($1::text[])"
)

model = None
user_item_matrix = None
user_lookup = None
product_lookup = None
user_id_to_idx = None
product_id_to_idx = None


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or DEFAULT_DATABASE_URL


def load_artifacts() -> None:
    global model, user_item_matrix, user_lookup, product_lookup
    global user_id_to_idx, product_id_to_idx

    model = AlternatingLeastSquares.load(str(ARTIFACTS_DIR / "als_model.npz"))
    user_item_matrix = scipy.sparse.load_npz(ARTIFACTS_DIR / "user_item_matrix.npz")
    with open(ARTIFACTS_DIR / "lookups.pkl", "rb") as lookups_file:
        lookups = pickle.load(lookups_file)
    user_lookup = lookups["user_lookup"]
    product_lookup = lookups["product_lookup"]
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_lookup)}
    product_id_to_idx = {str(product_id): idx for idx, product_id in enumerate(product_lookup)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    app.state.pool = await asyncpg.create_pool(database_url())
    yield
    await app.state.pool.close()


app = FastAPI(lifespan=lifespan)


async def hydrate_recommendations(
    pool: asyncpg.Pool,
    items: list[tuple[str, float]],
) -> list[ProductRecommendation]:
    asins = [asin for asin, _ in items]
    rows = await pool.fetch(METADATA_QUERY, asins)
    by_asin = {row["asin"]: row for row in rows}

    hydrated = []
    for asin, score in items:
        meta = by_asin.get(asin)
        hydrated.append(
            ProductRecommendation(
                product_id=asin,
                score=score,
                title=meta["title"] if meta else None,
                price=float(meta["price"]) if meta and meta["price"] is not None else None,
                im_url=meta["im_url"] if meta else None,
                brand=meta["brand"] if meta else None,
            )
        )
    return hydrated


@app.get("/recommend/{user_id}", response_model=list[ProductRecommendation])
async def recommend(
    user_id: int,
    k: int = 10,
    pool: asyncpg.Pool = Depends(get_db_pool),
):
    if 0 <= user_id < len(user_lookup):
        user_idx = user_id
    elif user_id in user_id_to_idx:
        user_idx = user_id_to_idx[user_id]
    else:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")

    item_ids, scores = model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=k,
        filter_already_liked_items=True,
    )

    ranked = [
        (str(product_lookup[int(item_idx)]), float(score))
        for item_idx, score in zip(np.asarray(item_ids), np.asarray(scores))
    ]
    return await hydrate_recommendations(pool, ranked)


@app.get("/similar/{product_id}", response_model=list[ProductRecommendation])
async def similar(
    product_id: str,
    k: int = 10,
    pool: asyncpg.Pool = Depends(get_db_pool),
):
    if product_id not in product_id_to_idx:
        raise HTTPException(status_code=404, detail=f"product_id {product_id} not found")

    item_idx = product_id_to_idx[product_id]
    similar_ids, scores = model.similar_items(itemid=item_idx, N=k + 1)

    ranked = []
    for similar_idx, score in zip(np.asarray(similar_ids), np.asarray(scores)):
        similar_idx = int(similar_idx)
        if similar_idx == item_idx:
            continue
        ranked.append((str(product_lookup[similar_idx]), float(score)))
        if len(ranked) >= k:
            break
    return await hydrate_recommendations(pool, ranked)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
