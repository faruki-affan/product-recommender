# E-Commerce Product Recommendation Engine

A production-grade machine learning recommendation pipeline engineered to deliver low-latency, personalized product suggestions. This system integrates an Alternating Least Squares (ALS) collaborative filtering model with a high-concurrency FastAPI backend, utilizing Redis for sub-millisecond cache retrieval and PostgreSQL for persistent state management. The entire architecture is fully containerized and supported by an automated continuous integration pipeline.

## System Architecture & Technology Stack

* **Machine Learning:** Implicit Alternating Least Squares (ALS) for collaborative filtering and matrix factorization.
* **API Layer:** FastAPI (Python 3.12, Uvicorn) for asynchronous, high-throughput request handling.
* **State & Storage:** PostgreSQL 16 (relational data, user profiles, interaction history) and Redis 7 (in-memory distributed caching).
* **DevOps & CI/CD:** Docker, Docker Compose, and GitHub Actions (automated `pytest` execution, container validation, ephemeral service provisioning).

## Core Engineering Features

* **Sub-Millisecond Retrieval:** Achieves near-instantaneous response times by pre-computing recommendations and serving them directly from a Redis cache, minimizing API latency.
* **Algorithmic Cold Start Handling:** Deploys adaptive fallback strategies (e.g., global trending products, category-based popularity algorithms) to serve new users lacking historical interaction data.
* **Automated Data Pipeline:** Includes modular scripts for continuous dataset ingestion, feature engineering, and automated ML model retraining.
* **Resilient Infrastructure:** Leverages Docker Compose for isolated microservices, featuring automated database health checks, persistent volume mounts, and strict container restart policies.
* **Continuous Integration:** Enforces code quality via GitHub Actions, automatically spinning up PostgreSQL and Redis service containers to execute the full integration test suite on every pull request.

## Local Deployment & Execution

The stack is fully containerized to guarantee environment parity across development, testing, and production.

**Prerequisites:** Docker Desktop installed and running.

**1. Clone the repository**

```bash
git clone https://github.com/faruki-affan/product-recommender.git
cd product-recommender

```

**2. Provision the infrastructure**

```bash
docker compose up --build -d

```

*This command compiles the API image, pulls required data stores, and initializes all microservices in the background.*

**3. Verify services & access the API**

```bash
docker compose ps

```

The interactive OpenAPI (Swagger) documentation is instantly accessible at: `http://localhost:8080/docs`

**4. Teardown**

```bash
docker compose down

```

*(Note: PostgreSQL state remains securely preserved in local Docker volumes post-teardown).*
