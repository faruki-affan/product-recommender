"""FastAPI service that serves ALS product recommendations from saved artifacts."""

import pickle
from pathlib import Path

import implicit  # noqa: F401
import numpy as np
import scipy.sparse
import uvicorn
from fastapi import FastAPI, HTTPException
from implicit.cpu.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

app = FastAPI()

model = None
user_item_matrix = None
user_lookup = None
product_lookup = None
user_id_to_idx = None
product_id_to_idx = None


@app.on_event("startup")
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


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
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

    return [
        {
            "product_id": str(product_lookup[int(item_idx)]),
            "score": float(score),
        }
        for item_idx, score in zip(np.asarray(item_ids), np.asarray(scores))
    ]


@app.get("/similar/{product_id}")
def similar(product_id: str, k: int = 10):
    if product_id not in product_id_to_idx:
        raise HTTPException(status_code=404, detail=f"product_id {product_id} not found")

    item_idx = product_id_to_idx[product_id]
    similar_ids, scores = model.similar_items(itemid=item_idx, N=k + 1)

    results = []
    for similar_idx, score in zip(np.asarray(similar_ids), np.asarray(scores)):
        similar_idx = int(similar_idx)
        if similar_idx == item_idx:
            continue
        results.append(
            {
                "product_id": str(product_lookup[similar_idx]),
                "score": float(score),
            }
        )
        if len(results) >= k:
            break
    return results


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
