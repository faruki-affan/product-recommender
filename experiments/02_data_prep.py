"""Download Amazon Video Games 5-core reviews and report real-dataset sparsity."""

from pathlib import Path
import ssl
import urllib.request

import pandas as pd

URL = (
    "https://jmcauley.ucsd.edu/data/amazon_v2/"
    "categoryFilesSmall/Video_Games_5.json.gz"
)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Video_Games_5.json.gz"

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
if DATA_PATH.exists():
    print(f"File already exists, skipping download: {DATA_PATH}")
else:
    print(f"Downloading {URL}")
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(URL, DATA_PATH)
    print(f"Saved to {DATA_PATH}")

reviews_df = pd.read_json(DATA_PATH, lines=True, compression="gzip")
reviews_df = reviews_df[["reviewerID", "asin", "overall"]].rename(
    columns={
        "reviewerID": "user_id",
        "asin": "product_id",
        "overall": "rating",
    }
)

n_rows = len(reviews_df)
n_users = reviews_df["user_id"].nunique()
n_products = reviews_df["product_id"].nunique()
sparsity_pct = 100 * (1 - n_rows / (n_users * n_products))

print(f"Total rows: {n_rows}")
print(f"Unique users: {n_users}")
print(f"Unique products: {n_products}")
print(f"Sparsity: {sparsity_pct:.2f}%")
