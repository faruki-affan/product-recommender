"""Bulk-load Amazon Video Games metadata into the products table."""

from __future__ import annotations

import ast
import gzip
import json
import os
import ssl
import urllib.request
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, execute_batch

from src.cache.client import invalidate_recommendation_cache_sync

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_METADATA_PATH = ROOT / "data" / "meta_Video_Games.json.gz"
METADATA_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "meta_Video_Games.json.gz"
)
DEFAULT_DATABASE_URL = "postgresql://postgres:9710@localhost:5432/recommender_db"
BATCH_SIZE = 2000

CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    asin TEXT PRIMARY KEY,
    title TEXT,
    price DOUBLE PRECISION,
    im_url TEXT,
    brand TEXT,
    categories JSONB,
    description TEXT
)
"""

INSERT_PRODUCTS = """
INSERT INTO products (asin, title, price, im_url, brand, categories, description)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (asin) DO UPDATE SET
    title = EXCLUDED.title,
    price = EXCLUDED.price,
    im_url = EXCLUDED.im_url,
    brand = EXCLUDED.brand,
    categories = EXCLUDED.categories,
    description = EXCLUDED.description
"""


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or DEFAULT_DATABASE_URL


def metadata_path() -> Path:
    raw = os.environ.get("METADATA_PATH")
    return Path(raw) if raw else DEFAULT_METADATA_PATH


def metadata_file_exists(path: Path) -> bool:
    return path.is_file()


def download_metadata(path: Path, url: str = METADATA_URL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")


def ensure_metadata_file(path: Path) -> Path:
    if metadata_file_exists(path):
        print(f"File already exists, skipping download: {path}")
        return path
    download_metadata(path)
    return path


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        record = ast.literal_eval(line)
    if not isinstance(record, dict):
        return None
    return record


def normalize_price(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_description(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        text = " ".join(str(part) for part in value if part)
        return text or None
    text = str(value).strip()
    return text or None


def to_row(record: dict) -> tuple | None:
    asin = record.get("asin")
    if not asin:
        return None
    categories = record.get("categories")
    return (
        str(asin),
        record.get("title"),
        normalize_price(record.get("price")),
        record.get("imUrl"),
        record.get("brand"),
        Json(categories) if categories is not None else None,
        normalize_description(record.get("description")),
    )


def iter_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = parse_line(line)
            if record is None:
                continue
            row = to_row(record)
            if row is not None:
                yield row


def load_metadata(path: Path, conn) -> int:
    inserted = 0
    batch: list[tuple] = []
    with conn.cursor() as cursor:
        cursor.execute(CREATE_PRODUCTS_TABLE)
        for row in iter_rows(path):
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                execute_batch(cursor, INSERT_PRODUCTS, batch, page_size=BATCH_SIZE)
                inserted += len(batch)
                batch.clear()
        if batch:
            execute_batch(cursor, INSERT_PRODUCTS, batch, page_size=BATCH_SIZE)
            inserted += len(batch)
    conn.commit()
    return inserted


def main() -> None:
    path = ensure_metadata_file(metadata_path())

    print(f"Loading metadata from {path}")
    with psycopg2.connect(database_url()) as conn:
        count = load_metadata(path, conn)
    print(f"Inserted/updated {count} products")
    try:
        deleted = invalidate_recommendation_cache_sync()
        print(f"Invalidated {deleted} recommendation cache keys")
    except Exception as exc:
        print(f"Warning: cache invalidation failed: {exc}")


if __name__ == "__main__":
    main()
