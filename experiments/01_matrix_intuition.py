"""Build a sparse user-item matrix from dummy interactions and report sparsity."""

import pandas as pd

# Dummy implicit feedback: 5 users, 6 products, 7 observed purchases.
interactions_df = pd.DataFrame(
    {
        "user_id": ["U1", "U1", "U2", "U3", "U3", "U4", "U5"],
        "product_id": ["P1", "P2", "P2", "P3", "P4", "P5", "P6"],
        "interactions": [1, 1, 1, 1, 1, 1, 1],
    }
)

user_item_matrix = pd.pivot_table(
    interactions_df,
    values="interactions",
    index="user_id",
    columns="product_id",
    fill_value=0,
)

n_users = user_item_matrix.shape[0]
n_products = user_item_matrix.shape[1]
n_observed = int(user_item_matrix.to_numpy().sum())
sparsity_pct = 100 * (1 - n_observed / (n_users * n_products))

print("Raw interactions DataFrame:")
print(interactions_df)
print()
print("User-Item Matrix:")
print(user_item_matrix)
print()
print(f"Sparsity: {sparsity_pct:.2f}%")
