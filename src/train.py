"""Train an implicit ALS model and persist artifacts for serving."""

import os
import pickle
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Video_Games_5.json.gz"
ARTIFACTS_DIR = ROOT / "artifacts"

print("Loading data...")
reviews_df = pd.read_json(DATA_PATH, lines=True, compression="gzip")
reviews_df = reviews_df[["reviewerID", "asin", "overall"]].rename(
    columns={
        "reviewerID": "user_id",
        "asin": "product_id",
        "overall": "rating",
    }
)

print("Mapping users and products...")
reviews_df["user_idx"], user_lookup = pd.factorize(reviews_df["user_id"])
reviews_df["product_idx"], product_lookup = pd.factorize(reviews_df["product_id"])

n_users = len(user_lookup)
n_items = len(product_lookup)

user_item_matrix = csr_matrix(
    (
        np.ones(len(reviews_df), dtype=np.float32),
        (reviews_df["user_idx"].to_numpy(), reviews_df["product_idx"].to_numpy()),
    ),
    shape=(n_users, n_items),
)
# Keep implicit feedback binary even if a user-item pair appears more than once.
user_item_matrix.data[:] = 1.0

print("Training model...")
model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=20,
    random_state=42,
)
model.fit(user_item_matrix)

print("Saving artifacts...")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
model.save("artifacts/als_model.npz")
scipy.sparse.save_npz("artifacts/user_item_matrix.npz", user_item_matrix)
with open("artifacts/lookups.pkl", "wb") as lookups_file:
    pickle.dump(
        {"user_lookup": user_lookup, "product_lookup": product_lookup},
        lookups_file,
    )

print("Done. Artifacts saved to artifacts/")
