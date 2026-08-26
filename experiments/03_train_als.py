"""Train an implicit ALS model on Amazon Video Games reviews and print sample recs."""

from pathlib import Path

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Video_Games_5.json.gz"

reviews_df = pd.read_json(DATA_PATH, lines=True, compression="gzip")
reviews_df = reviews_df[["reviewerID", "asin", "overall"]].rename(
    columns={
        "reviewerID": "user_id",
        "asin": "product_id",
        "overall": "rating",
    }
)

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

model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=20,
    random_state=42,
)
model.fit(user_item_matrix)

user_idx = 0
user_id = user_lookup[user_idx]
liked_product_ids = product_lookup[
    user_item_matrix[user_idx].indices[:5]
]

ids, scores = model.recommend(
    0,
    user_item_matrix[0],
    N=10,
    filter_already_liked_items=True,
)

print(f"User ID: {user_id}")
print("Already interacted with (5 games):")
for product_id in liked_product_ids:
    print(f"  {product_id}")

print("Top 10 recommended games:")
for product_idx, score in zip(ids, scores):
    print(f"  {product_lookup[product_idx]}\t{score:.6f}")
