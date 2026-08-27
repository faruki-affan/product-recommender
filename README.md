# 🛒 E-Commerce Product Recommendation Engine

An end-to-end, production-ready machine learning recommendation system built to deliver real-time, personalized product suggestions. This project integrates a collaborative filtering machine learning model with a high-performance API, persistent data storage, and caching, all orchestrated via Docker and automated with GitHub Actions.

## 🏗️ System Architecture & Tech Stack

* **Machine Learning:** Implicit Alternating Least Squares (ALS) for collaborative filtering.
* **Backend API:** FastAPI (Python 3.12) for asynchronous, high-throughput endpoint routing.
* **Database:** PostgreSQL 16 for persistent storage of user profiles, product metadata, and interaction history.
* **Cache:** Redis 7 for sub-millisecond retrieval of pre-computed recommendations.
* **Containerization:** Docker & Docker Compose for isolated, multi-service deployment.
* **CI/CD:** GitHub Actions for automated unit testing (`pytest`) and Docker build verification.

## ✨ Key Features

* **Real-Time Recommendations:** Fetches user-specific product suggestions in milliseconds using Redis caching.
* **Cold Start Handling:** Implements fallback strategies for new users without prior interaction history.
* **Automated Data Pipeline:** Scripts for dataset ingestion, preprocessing, and automated ML model retraining.
* **Resilient Infrastructure:** Configured with database health checks, volume persistence, and automatic container restarts.
* **Continuous Integration:** Every pull request triggers a virtual environment to build the Docker image and execute the test suite against live PostgreSQL and Redis service containers.

## 🚀 Local Deployment (Docker)

The entire stack is containerized for seamless execution across any environment.

**Prerequisites:** 
* Docker Desktop installed and running.

**1. Clone the repository**
```bash
git clone [https://github.com/faruki-affan/product-recommender.git](https://github.com/faruki-affan/product-recommender.git)
cd product-recommender
