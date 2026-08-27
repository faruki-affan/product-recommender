"""Tests for ALS inference, artifact loading, and embedding lookups."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.sparse
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from tests.conftest import FakeALSModel, install_test_artifacts


def _tiny_interaction_matrix() -> csr_matrix:
    return csr_matrix(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def fitted_als():
    matrix = _tiny_interaction_matrix()
    model = AlternatingLeastSquares(
        factors=8,
        regularization=0.1,
        iterations=8,
        random_state=42,
    )
    model.fit(matrix)
    return model, matrix


def test_als_recommend_returns_ranked_unseen_items(fitted_als):
    model, matrix = fitted_als
    user_idx = 0
    item_ids, scores = model.recommend(
        user_idx,
        matrix[user_idx],
        N=2,
        filter_already_liked_items=True,
    )

    liked = set(matrix[user_idx].indices)
    assert len(item_ids) == 2
    assert len(scores) == 2
    assert np.all(np.diff(scores) <= 0)
    assert liked.isdisjoint(set(np.asarray(item_ids).tolist()))


def test_als_similar_items_includes_query_then_neighbors(fitted_als):
    model, _matrix = fitted_als
    item_idx = 0
    similar_ids, scores = model.similar_items(itemid=item_idx, N=3)

    assert int(similar_ids[0]) == item_idx
    assert len(similar_ids) == 3
    assert pytest.approx(float(scores[0]), rel=1e-3) == 1.0 or scores[0] >= scores[1]


def test_embedding_factor_shapes(fitted_als):
    model, matrix = fitted_als
    n_users, n_items = matrix.shape

    assert model.user_factors.shape[0] == n_users
    assert model.item_factors.shape[0] == n_items
    assert model.user_factors.shape[1] == model.item_factors.shape[1] == 8


def test_embedding_dot_product_ranks_like_recommend(fitted_als):
    model, matrix = fitted_als
    user_idx = 1
    user_vec = model.user_factors[user_idx]
    scores = model.item_factors @ user_vec
    scores[matrix[user_idx].indices] = -np.inf
    top_from_dot = int(np.argmax(scores))

    rec_ids, _ = model.recommend(
        user_idx,
        matrix[user_idx],
        N=1,
        filter_already_liked_items=True,
    )
    assert int(rec_ids[0]) == top_from_dot


def test_id_lookups_map_raw_ids_and_matrix_indices():
    install_test_artifacts()
    import src.api.main as main

    assert main.user_id_to_idx[100] == 0
    assert main.user_id_to_idx[200] == 1
    assert main.product_id_to_idx["B001"] == 0
    assert main.product_id_to_idx["B002"] == 1
    assert 0 <= 0 < len(main.user_lookup)
    assert "MISSING" not in main.product_id_to_idx


def test_fake_model_similar_items_skips_self_when_ranking():
    model = FakeALSModel(n_items=2)
    similar_ids, _scores = model.similar_items(itemid=0, N=3)
    ranked = [int(idx) for idx in similar_ids if int(idx) != 0]
    assert ranked[0] == 1


def test_load_artifacts_restores_model_matrix_and_lookups(tmp_path, monkeypatch):
    matrix = _tiny_interaction_matrix()
    model = AlternatingLeastSquares(
        factors=4,
        regularization=0.1,
        iterations=4,
        random_state=42,
    )
    model.fit(matrix)

    model.save(str(tmp_path / "als_model.npz"))
    scipy.sparse.save_npz(tmp_path / "user_item_matrix.npz", matrix)
    with open(tmp_path / "lookups.pkl", "wb") as handle:
        pickle.dump(
            {
                "user_lookup": np.array([10, 20, 30]),
                "product_lookup": np.array(["A", "B", "C", "D"]),
            },
            handle,
        )

    import src.api.main as main

    monkeypatch.setattr(main, "ARTIFACTS_DIR", tmp_path)
    main.load_artifacts()

    assert main.user_item_matrix.shape == (3, 4)
    assert list(main.user_lookup) == [10, 20, 30]
    assert main.user_id_to_idx[20] == 1
    assert main.product_id_to_idx["C"] == 2

    item_ids, scores = main.model.recommend(0, main.user_item_matrix[0], N=2)
    assert len(item_ids) == 2
    assert len(scores) == 2
