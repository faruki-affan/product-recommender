"""Pydantic response models for the recommendation API."""

from pydantic import BaseModel


class ProductRecommendation(BaseModel):
    product_id: str
    score: float
    title: str | None = None
    price: float | None = None
    im_url: str | None = None
    brand: str | None = None
