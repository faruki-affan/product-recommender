"""Tests for database URL helpers, session usage, and metadata loaders."""

from __future__ import annotations

import gzip
import json
from unittest.mock import MagicMock, patch

from psycopg2.extras import Json

from src.api.main import database_url as api_database_url
from src.db.load_metadata import (
    CREATE_PRODUCTS_TABLE,
    database_url,
    ensure_metadata_file,
    iter_rows,
    load_metadata,
    metadata_file_exists,
    metadata_path,
    normalize_description,
    normalize_price,
    parse_line,
    to_row,
)


def test_database_url_prefers_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-database-url/db")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://from-postgres-url/db")

    assert database_url() == "postgresql://from-database-url/db"
    assert api_database_url() == "postgresql://from-database-url/db"


def test_database_url_falls_back_to_postgres_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_URL", "postgresql://from-postgres-url/db")

    assert database_url() == "postgresql://from-postgres-url/db"


def test_metadata_path_uses_env_override(monkeypatch, tmp_path):
    override = tmp_path / "meta.json.gz"
    monkeypatch.setenv("METADATA_PATH", str(override))

    assert metadata_path() == override


def test_metadata_file_exists(tmp_path):
    missing = tmp_path / "missing.json.gz"
    present = tmp_path / "present.json.gz"
    present.write_bytes(b"x")

    assert metadata_file_exists(missing) is False
    assert metadata_file_exists(present) is True


def test_ensure_metadata_file_skips_download_when_present(tmp_path):
    path = tmp_path / "meta.json.gz"
    path.write_bytes(b"gz")

    with patch("src.db.load_metadata.download_metadata") as download:
        result = ensure_metadata_file(path)

    assert result == path
    download.assert_not_called()


def test_parse_line_json_and_python_literal():
    json_record = parse_line('{"asin": "B001", "title": "Halo"}')
    literal_record = parse_line("{'asin': 'B002', 'title': 'Mario'}")

    assert json_record == {"asin": "B001", "title": "Halo"}
    assert literal_record == {"asin": "B002", "title": "Mario"}
    assert parse_line("   ") is None
    assert parse_line("[]") is None


def test_normalize_price_and_description():
    assert normalize_price("19.99") == 19.99
    assert normalize_price("") is None
    assert normalize_price("free") is None
    assert normalize_description(["fun", "game"]) == "fun game"
    assert normalize_description("  boxed  ") == "boxed"
    assert normalize_description([]) is None
    assert normalize_description(None) is None


def test_to_row_requires_asin_and_wraps_categories():
    assert to_row({"title": "No asin"}) is None

    row = to_row(
        {
            "asin": "B00TEST",
            "title": "Test Game",
            "price": "12.5",
            "imUrl": "http://img",
            "brand": "Acme",
            "categories": [["Games"]],
            "description": "A game",
        }
    )

    assert row[0] == "B00TEST"
    assert row[2] == 12.5
    assert row[3] == "http://img"
    assert isinstance(row[5], Json)


def test_iter_rows_reads_gzip_metadata(tmp_path):
    path = tmp_path / "meta.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"asin": "B001", "title": "One", "price": 1}) + "\n")
        handle.write("\n")
        handle.write(json.dumps({"title": "skipped"}) + "\n")
        handle.write(json.dumps({"asin": "B002", "title": "Two"}) + "\n")

    rows = list(iter_rows(path))

    assert [row[0] for row in rows] == ["B001", "B002"]
    assert rows[0][1] == "One"


def test_load_metadata_creates_table_batches_and_commits(tmp_path):
    path = tmp_path / "meta.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"asin": "B001", "title": "One"}) + "\n")
        handle.write(json.dumps({"asin": "B002", "title": "Two"}) + "\n")

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("src.db.load_metadata.execute_batch") as execute_batch:
        inserted = load_metadata(path, conn)

    cursor.execute.assert_called_once_with(CREATE_PRODUCTS_TABLE)
    execute_batch.assert_called_once()
    assert execute_batch.call_args[0][0] is cursor
    assert inserted == 2
    conn.commit.assert_called_once()
