"""Locust load tests for the recommendation API."""

from locust import HttpUser, between, task


class RecommenderUser(HttpUser):
    """Simulated client hitting recommendation and similarity endpoints."""

    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"

    @task
    def recommend(self) -> None:
        self.client.get("/recommend/0?k=10", name="/recommend/{user_id}")

    @task
    def similar(self) -> None:
        self.client.get("/similar/B0053BCML6?k=10", name="/similar/{product_id}")
