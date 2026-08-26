"""Leave-one-out temporal evaluation of a freshly trained implicit ALS model."""

from pathlib import Path

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Video_Games_5.json.gz"

ALS_FACTORS = 64
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 20
ALS_RANDOM_STATE = 42
EVAL_KS = (5, 10, 20)


def load_interactions() -> pd.DataFrame:
    print(f"Loading {DATA_PATH}")
    reviews_df = pd.read_json(DATA_PATH, lines=True, compression="gzip")
    reviews_df = reviews_df[
        ["reviewerID", "asin", "overall", "unixReviewTime"]
    ].rename(
        columns={
            "reviewerID": "user_id",
            "asin": "product_id",
            "overall": "rating",
            "unixReviewTime": "unixReviewTime",
        }
    )
    print(f"Loaded {len(reviews_df):,} interactions")
    return reviews_df


def leave_one_out_temporal_split(reviews_df: pd.DataFrame):
    counts = reviews_df.groupby("user_id").size()
    eligible_user_ids = counts[counts >= 2].index
    n_users_ge2 = len(eligible_user_ids)

    eligible_df = reviews_df[reviews_df["user_id"].isin(eligible_user_ids)].copy()
    eligible_df = eligible_df.sort_values(
        ["user_id", "unixReviewTime"], kind="mergesort"
    )

    test_df = eligible_df.groupby("user_id", as_index=False).tail(1)
    train_df = eligible_df.drop(test_df.index)

    overlap = train_df.merge(
        test_df[["user_id", "product_id"]],
        on=["user_id", "product_id"],
        how="inner",
    )
    n_overlap_rows = len(overlap)
    if n_overlap_rows:
        print(
            f"Removing {n_overlap_rows:,} training rows that duplicate "
            "held-out (user_id, product_id) pairs"
        )
        train_df = train_df.merge(
            test_df[["user_id", "product_id"]],
            on=["user_id", "product_id"],
            how="left",
            indicator=True,
        )
        train_df = train_df[train_df["_merge"] == "left_only"].drop(columns=["_merge"])

    users_with_train = set(train_df["user_id"].unique())
    n_empty_train = n_users_ge2 - len(users_with_train)
    test_df = test_df[test_df["user_id"].isin(users_with_train)].copy()

    leftover = train_df.merge(
        test_df[["user_id", "product_id"]],
        on=["user_id", "product_id"],
        how="inner",
    )
    assert leftover.empty, "Test (user, product) pairs leaked into training data"

    print(f"Users with >=2 interactions: {n_users_ge2:,}")
    print(f"Training interactions: {len(train_df):,}")
    print(f"Held-out test interactions: {len(test_df):,}")
    if n_empty_train:
        print(
            f"Excluded users (no remaining train interactions after overlap removal): "
            f"{n_empty_train:,}"
        )

    return train_df, test_df, n_users_ge2, n_empty_train


def build_train_matrix(train_df: pd.DataFrame):
    train_df = train_df.copy()
    train_df["user_idx"], user_lookup = pd.factorize(train_df["user_id"])
    train_df["product_idx"], product_lookup = pd.factorize(train_df["product_id"])

    n_users = len(user_lookup)
    n_items = len(product_lookup)
    user_item_matrix = csr_matrix(
        (
            np.ones(len(train_df), dtype=np.float32),
            (train_df["user_idx"].to_numpy(), train_df["product_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    user_item_matrix.data[:] = 1.0

    print(
        f"Training matrix: {n_users:,} users x {n_items:,} items, "
        f"{user_item_matrix.nnz:,} observed pairs"
    )
    return train_df, user_item_matrix, user_lookup, product_lookup


def select_evaluation_users(test_df: pd.DataFrame, user_lookup, product_lookup):
    user_to_idx = {user_id: i for i, user_id in enumerate(user_lookup)}
    product_to_idx = {product_id: i for i, product_id in enumerate(product_lookup)}

    eval_df = test_df.copy()
    eval_df["user_idx"] = eval_df["user_id"].map(user_to_idx)
    eval_df["product_idx"] = eval_df["product_id"].map(product_to_idx)

    unseen_mask = eval_df["product_idx"].isna()
    n_excluded_unseen = int(unseen_mask.sum())
    eval_df = eval_df.loc[~unseen_mask].copy()
    eval_df["user_idx"] = eval_df["user_idx"].astype(np.int32)
    eval_df["product_idx"] = eval_df["product_idx"].astype(np.int32)

    assert eval_df["user_idx"].notna().all(), "Evaluation user missing from training map"

    print(f"Excluded users (test item unseen in training): {n_excluded_unseen:,}")
    print(f"Evaluation users: {len(eval_df):,}")
    return eval_df, n_excluded_unseen


def train_als(user_item_matrix: csr_matrix):
    print(
        "Training a new ALS model on training interactions only "
        f"(factors={ALS_FACTORS}, regularization={ALS_REGULARIZATION}, "
        f"iterations={ALS_ITERATIONS}, random_state={ALS_RANDOM_STATE})"
    )
    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=ALS_RANDOM_STATE,
    )
    model.fit(user_item_matrix)
    return model


def assert_no_train_test_row_overlap(user_item_matrix: csr_matrix, eval_df: pd.DataFrame):
    for user_idx, product_idx in zip(
        eval_df["user_idx"].to_numpy(), eval_df["product_idx"].to_numpy()
    ):
        liked = user_item_matrix[user_idx].indices
        if product_idx in liked:
            raise AssertionError(
                f"Held-out item {product_idx} is present in training row for user {user_idx}"
            )


def evaluate(model, user_item_matrix: csr_matrix, eval_df: pd.DataFrame):
    user_indices = eval_df["user_idx"].to_numpy()
    test_items = eval_df["product_idx"].to_numpy()
    max_k = max(EVAL_KS)

    print(f"Generating Top-{max_k} recommendations for {len(user_indices):,} users...")
    recommended_ids, _scores = model.recommend(
        user_indices,
        user_item_matrix[user_indices],
        N=max_k,
        filter_already_liked_items=True,
    )

    print("Checking that training items are filtered and test items stay eligible...")
    for i, user_idx in enumerate(user_indices):
        liked = set(user_item_matrix[user_idx].indices)
        recs = recommended_ids[i]
        if liked.intersection(recs):
            raise AssertionError(
                f"Recommended items include already-liked training items for user {user_idx}"
            )
        if test_items[i] in liked:
            raise AssertionError(
                f"Held-out test item remains in the training row for user {user_idx}"
            )

    print("Computing Precision@K / Recall@K / Hit Rate@K...")
    metrics = {}
    for k in EVAL_KS:
        hits = (recommended_ids[:, :k] == test_items[:, None]).any(axis=1)
        hit_rate = float(hits.mean())
        recall = hit_rate
        precision = float((hits.astype(np.float64) / k).mean())
        metrics[k] = {
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
        }
    return metrics


def print_report(n_users_ge2, n_eval, n_excluded_unseen, n_empty_train, metrics):
    print()
    print("========================================")
    print("Phase 4 - Offline Evaluation")
    print("========================================")
    print()
    print(f"Users with >=2 interactions: {n_users_ge2}")
    print(f"Evaluation users: {n_eval}")
    print(f"Excluded users (test item unseen in training): {n_excluded_unseen}")
    print(
        f"Excluded users (no remaining train interactions after overlap removal): "
        f"{n_empty_train}"
    )
    print()
    print("ALS Parameters:")
    print(f"Factors: {ALS_FACTORS}")
    print(f"Regularization: {ALS_REGULARIZATION}")
    print(f"Iterations: {ALS_ITERATIONS}")
    print(f"Random State: {ALS_RANDOM_STATE}")
    print()
    print("----------------------------------------")
    print("K       Precision      Recall      Hit Rate")
    print("----------------------------------------")
    for k in EVAL_KS:
        m = metrics[k]
        print(
            f"{k:<8}{m['precision']:.3f}          {m['recall']:.3f}       {m['hit_rate']:.3f}"
        )
    print("----------------------------------------")


def main():
    # Phase 4 trains a new model from the training split only.
    # The Phase 3 full-data model is never loaded or reused.
    reviews_df = load_interactions()
    train_df, test_df, n_users_ge2, n_empty_train = leave_one_out_temporal_split(
        reviews_df
    )
    train_df, user_item_matrix, user_lookup, product_lookup = build_train_matrix(
        train_df
    )
    eval_df, n_excluded_unseen = select_evaluation_users(
        test_df, user_lookup, product_lookup
    )
    assert_no_train_test_row_overlap(user_item_matrix, eval_df)

    model = train_als(user_item_matrix)
    metrics = evaluate(model, user_item_matrix, eval_df)
    print_report(
        n_users_ge2, len(eval_df), n_excluded_unseen, n_empty_train, metrics
    )


if __name__ == "__main__":
    main()
