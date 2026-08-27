# Product Recommendation System - Project Report

## Phase 0: Project Environment Setup & Tooling Configuration
**Status:** Completed

**What I Built:**
* Established a structured project workspace with dedicated directories for data, experiments, core application logic, and models.
* Created a `.gitignore` file to ensure large ML data binaries and secret files are kept out of version control.
* Set up an isolated Python virtual environment (`.venv`) to prevent system-level package conflicts.
* Defined and installed project dependencies in `requirements.txt` (FastAPI, Pandas, NumPy, Implicit, PostgreSQL drivers, Redis, etc.).
* Wrote an environment verification script (`00_env_check.py`) to confirm that C-extensions for heavy ML libraries (like SciPy) installed correctly and work seamlessly alongside the asynchronous backend framework.

**Key Technical Decisions:**
* **Separation of Concerns:** Kept model artifacts and datasets strictly out of Git to maintain a clean, lightweight repository history.
* **Pre-compiled Binaries:** Used `psycopg2-binary` for rapid local development without needing local C-compilers.

## Phase 1: Understand Recommendation Systems
**Status:** Completed

**What I Built:**
* A matrix intuition script (`experiments/01_matrix_intuition.py`) to simulate user-item interactions.
* Pivoted raw transaction data into a dense User-Item Matrix.
* Implemented a sparsity calculation function.

**Key Concepts Learned:**
* **Collaborative Filtering vs. Content-Based:** We are using Collaborative Filtering to recommend based on behavioral overlap rather than item metadata.
* **Implicit Feedback:** We treat interactions (like purchases or clicks) as binary signals (1 for observed, 0 for unobserved).
* **Sparsity:** Real-world interaction matrices are >99% empty. Understanding sparsity is critical because it dictates why we must use memory-efficient data structures (like `scipy.sparse` CSR matrices) and mathematical techniques (like Matrix Factorization) to predict the missing 0s.

## Phase 2: Dataset Ingestion & Preprocessing
**Status:** Completed

**What I Built:**
* Data ingestion pipeline in `experiments/02_data_prep.py` that downloads the Amazon Video Games 5-core dataset (JSONL gzip).
* Preprocessing logic that extracts user IDs, product IDs, and interaction ratings into structured Pandas DataFrames.
* Calculated real-world catalog sparsity metrics.

**Dataset Statistics:**
* Total Interactions: 497,577
* Unique Users: 55,223
* Unique Items: 17,408
* Matrix Sparsity: 99.95%

**Key Architectural Insight:**
* Standard dense 2D matrices scale as $O(M \times N)$, consuming ~8GB of memory for 1B cells. 
* A sparse matrix (CSR format) scales as $O(\text{interactions})$, storing only the non-zero cells and reducing memory consumption down to ~10MB.


## Phase 3: Baseline Matrix Factorization & Collaborative Filtering

**Status:** Completed  
**Script:** `experiments/03_train_als.py`

---

### 1. Overview & Objectives
Phase 3 establishes the core Machine Learning baseline for the product recommendation engine. The objective is to build an implicit Collaborative Filtering system using **Matrix Factorization via Alternating Least Squares (iALS)** to learn latent representations of users and products from sparse interaction data.

---

### 2. Theoretical & Mathematical Foundation

#### A. Collaborative Filtering vs. Content-Based
* **Collaborative Filtering:** Operates strictly on user-item interaction behavior. It requires no item metadata (such as titles, descriptions, or genres) and recommends items based on patterns of co-interaction across users.

#### B. Implicit Feedback Formulation
* In implicit feedback scenarios (clicks, purchases, views), unobserved entries ($0$) do not indicate negative sentiment; they indicate a lack of exposure.
* Binary interaction confidence is defined as:
  $$p_{u, i} = \begin{cases} 1 & \text{if interaction exists} \\ 0 & \text{otherwise} \end{cases}$$

#### C. Matrix Factorization (Latent Factor Model)
The goal is to approximate the high-dimensional, sparse interaction matrix $R \in \mathbb{R}^{M \times N}$ by decomposing it into two dense, low-rank matrices:
* **User Embedding Matrix:** $U \in \mathbb{R}^{M \times d}$ (where each row $u_u$ is a $d$-dimensional user profile vector)
* **Item Embedding Matrix:** $V \in \mathbb{R}^{N \times d}$ (where each row $v_i$ is a $d$-dimensional product profile vector)

$$\hat{R} \approx U V^T$$

The predicted affinity score for User $u$ and Item $i$ is calculated via the inner dot product of their latent vectors:
$$\hat{r}_{u, i} = u_u \cdot v_i^T = \sum_{f=1}^{d} U_{u, f} V_{i, f}$$

The higher the dot product score, the stronger the recommendation.

---

### 3. Implementation Workflow

The pipeline implemented in `experiments/03_train_als.py` executes the following steps:

```text
Raw Interaction Data (JSONL)
         │
         ▼
Two-Way Integer ID Mapping (pd.factorize)
         │
         ▼
Compressed Sparse Row (CSR) Matrix Construction
         │
         ▼
Implicit Alternating Least Squares (iALS) Optimization
         │
         ▼
Top-K Recommendation Inference (with historical interaction masking)
         │
         ▼
Index-to-String ID Decoding

```

## Phase 4: Offline Evaluation & Metrics (Hit Rate@K)

**Status:** Completed  
**Script:** `experiments/04_offline_eval.py`

---

### 1. Objective
To rigorously test the recommendation model's accuracy on unseen data using industry-standard offline evaluation metrics, ensuring the model generalizes well to future user behavior rather than just memorizing the past.

### 2. Evaluation Methodology (Temporal Leave-One-Out)
To prevent data leakage (the "time-travel" problem), the dataset was split chronologically rather than randomly:
* **Training Set:** All historical user interactions *except* their most recent one.
* **Test Set (The Target):** The single most recent interaction for every user with at least 2 reviews.
* **Leakage Guards:** Strict isolation between training and testing data. If a user reviewed a game multiple times and it was chosen as the test target, all historical duplicates of that game were purged from their training matrix to prevent the model from cheating.

### 3. Results (Evaluated on 55,165 Users)

| Metric | K = 5 | K = 10 | K = 20 |
| :--- | :--- | :--- | :--- |
| **Hit Rate (HR@K)** | 3.0% (0.030) | 4.5% (0.045) | 6.9% (0.069) |
| **Precision@K** | 0.006 | 0.005 | 0.003 |
| **Recall@K** | 0.030 | 0.045 | 0.069 |

*(Note: Because there is exactly one target item per user, Recall@K is mathematically identical to Hit Rate@K).*

### 4. Performance Context: Is 4.5% Good?
In traditional Machine Learning classifications, 4.5% accuracy is failing. In Recommendation Systems, this represents a highly predictive model:
* **The Random Baseline:** Guessing a user's exact next purchase at random from a catalog of 17,408 unique items yields a baseline Top-10 Hit Rate of **~0.057%**.
* **The Performance Gain:** The iALS model achieves **4.5%**, performing roughly **80 times better** than random chance.
* **Strict Penalties:** Offline evaluation is extremely strict. If the model recommends 10 highly relevant, personalized games that the user would love, but misses the *one* specific game the user actually bought next, the model scores a 0. Therefore, a 4.5% exact Hit Rate proves the model has successfully built a robust, personalized latent space.


# Phase 5: Production Architecture, Decoupling, & Web Serving

## Architectural Overview

Phase 5 transitions the system from standalone experimental scripts (`experiments/`) into a decoupled, production-grade Machine Learning microservice. The offline batch job (`src/train.py`) processes raw Amazon 5-core JSON data to serialize three deterministic artifacts: `als_model.npz` (factor weights), `user_item_matrix.npz` (sparse CSR matrix), and `lookups.pkl` (bi-directional ID maps). Upon startup, the FastAPI online engine (`src/api/main.py`) loads these pre-computed artifacts directly into RAM, enabling real-time serving via `GET /recommend/{user_id}` (personalized dot-product ranking) and `GET /similar/{product_id}` (latent cosine item neighbors).

## 1. System Decoupling Rationale

In real-time production systems, offline training and online serving remain strictly decoupled to guarantee low-latency inference and high availability:

* **Separation of Compute Loads**: Processing 500k+ interactions requires intensive CPU and memory allocation over multi-second or multi-minute execution windows, whereas online recommendation serving must complete within single-digit milliseconds (< 15 ms).
* **Zero-Downtime Serving**: The web server never directly interacts with or processes the raw dataset; it loads pre-calculated mathematical factors into memory at startup and handles incoming HTTP traffic concurrently.
* **Idempotent Artifacts**: The training process produces deterministic binary artifacts saved to `artifacts/`, allowing the production API to consume them strictly in read-only mode.

## 2. Artifact Schema & Persistence

The offline training pipeline (`src/train.py`) outputs three core binary artifacts:

* `als_model.npz` (NumPy Compressed Archive, ~5–10 MB): Stores the learned user factor matrix $X \in \mathbb{R}^{M \times f}$ and item factor matrix $Y \in \mathbb{R}^{N \times f}$ ($f=64$ latent factors).
* `user_item_matrix.npz` (SciPy Sparse CSR, ~2–4 MB): Preserves historical user-item interactions to filter out already-consumed products dynamically during inference via `filter_already_liked_items=True`.
* `lookups.pkl` (Pickle Dictionary, ~1–2 MB): Stores raw Amazon ASIN and User ID lookup arrays inverted into $O(1)$ mapping dictionaries (`user_id_to_idx` and `product_id_to_idx`).

## 3. API Endpoints & Mathematical Mechanics

### 1. Personalized User Recommendations

* **Route**: `GET /recommend/{user_id}?k=10`
* **Status Codes**: `200 OK`, `404 Not Found` (Unknown User), `422 Validation Error` (Malformed input parameters)
* **Mechanics**: Computes the dot product between the target user factor vector $u_i$ and candidate item vectors $v_j$: $\text{score}_{ij} = u_i \cdot v_j^T$. The top $k$ items by score are ranked and returned after filtering historical interactions.
* **Sample Request**: `curl -X GET "[http://127.0.0.1:8000/recommend/0?k=3](http://127.0.0.1:8000/recommend/0?k=3)"`
* **Sample Response**: `[{"product_id": "B0053BCML6", "score": 0.2095499038696289}, {"product_id": "B000ZKA0J6", "score": 0.16577526926994324}, {"product_id": "B000FQ2DTA", "score": 0.12512291967868805}]`

### 2. Item-to-Item Similarities (Related Products)

* **Route**: `GET /similar/{product_id}?k=10`
* **Status Codes**: `200 OK`, `404 Not Found` (Unknown Product ID)
* **Mechanics**: Computes the cosine similarity across latent item vectors: $\text{Cosine Similarity}(v_a, v_b) = \frac{v_a \cdot v_b}{\Vert{}v_a\Vert{}_2 \Vert{}v_b\Vert{}_2}$. The system requests $N = k + 1$ candidates to drop the queried item (self-similarity of 1.0) and returns the top $k$ nearest neighbors.
* **Sample Request**: `curl -X GET "[http://127.0.0.1:8000/similar/B0053BCML6?k=3](http://127.0.0.1:8000/similar/B0053BCML6?k=3)"`
* **Sample Response**: `[{"product_id": "B000XJD33E", "score": 0.9473585486412048}, {"product_id": "B007AP8RJ4", "score": 0.9402635097503662}, {"product_id": "B001FVSOQ0", "score": 0.9363707304000854}]`

## 4. Execution & Verification Workflow

* **Step 1: Execute Offline Pipeline**: Run `python src/train.py` to generate `artifacts/als_model.npz`, `artifacts/user_item_matrix.npz`, and `artifacts/lookups.pkl`.
* **Step 2: Start Web Service**: Run `python src/api/main.py` to launch the Uvicorn ASGI server locally on `[http://127.0.0.1:8000](http://127.0.0.1:8000)`.
* **Step 3: Interactive Verification**: Navigate to `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` to test endpoints and validate payloads via the interactive Swagger UI interface.

## Phase 6: PostgreSQL Schema Design, Indexing, & Metadata Hydration

This phase establishes the relational catalog storage layer, streaming unstructured JSON metadata from the Amazon Video Games dataset directly into PostgreSQL, and hydrating the machine learning recommendation pipeline with rich product details.

### What Was Built

* **Database Schema Design**: Created a structured `products` table in PostgreSQL using `JSONB` for flexible category handling and explicit column mappings for identifiers, titles, pricing, brand names, and image URLs.
* **Streaming ETL Loader (`src/db/load_metadata.py`)**: Built a robust, memory-efficient ingestion script that streams and unzips compressed `.json.gz` catalog archives, normalizes unstructured attributes, and batch-inserts records into PostgreSQL using `psycopg2.extras.execute_batch` with upsert support.
* **API Metadata Hydration (`src/api/main.py`)**: Integrated asynchronous database connection pooling via `asyncpg` directly into the FastAPI application lifespan, enabling `/recommend` and `/similar` endpoints to map raw recommendation ASIN arrays to live catalog fields dynamically.

### Key Components & Architecture

* **Database Table (`products`)**:
* `asin` (TEXT, Primary Key)
* `title` (TEXT)
* `price` (DOUBLE PRECISION)
* `im_url` (TEXT)
* `brand` (TEXT)
* `categories` (JSONB)
* `description` (TEXT)


* **Hydration Workflow**: ML model inference returns top-K ranked `asin` scores, which are subsequently queried against the PostgreSQL catalog in a single batch query (`WHERE asin = ANY($1::text[])`) to return complete product attributes to clients.


## Phase 7: FastAPI Serving Layer, Schemas, & Dependency Injection

This phase refactors the application's serving layer to adopt professional software engineering patterns, separating concerns into dedicated modules, explicit Pydantic schemas, and FastAPI's dependency injection system.

### What Was Built

* **Modular Pydantic Schemas (`src/api/schemas.py`)**: Extracted data validation and response contracts into dedicated models (`ProductRecommendation`), ensuring strict typing for product identifiers, scores, prices, titles, image URLs, and brands.
* **Dependency Injection (`src/api/deps.py`)**: Implemented a reusable asynchronous database dependency (`get_db_pool()`) to supply the active `asyncpg` connection pool via FastAPI's native `Depends` system, removing reliance on global application state inside path operations.
* **Refactored Path Operations (`src/api/main.py`)**: Cleaned up route handlers to leverage injected database connections and strictly enforce response models, resulting in an easily testable and maintainable architecture.

### Key Components & Architecture

* **Schemas (`ProductRecommendation`)**: Enforces explicit data structures for all outgoing recommendation and similarity payloads.
* **Dependency Injection (`get_db_pool`)**: Standardizes how HTTP endpoints request database access resources.
* **Clean Routing**: Separates business logic and data mapping from global setup configurations.


## Phase 8: Redis In-Memory Caching & Cache Invalidation Strategy

This phase integrates an asynchronous Redis in-memory caching layer into the FastAPI application to dramatically reduce response times for frequent recommendation and similarity queries.

### What Was Built

* **Asynchronous Redis Client (`src/cache/client.py`)**: Configured a high-performance connection pool targeting `localhost:6379`, integrated natively into FastAPI's application lifespan events (`startup` and `shutdown`).
* **Endpoint-Level Caching**: Wrapped `/recommend/{user_id}` and `/similar/{product_id}` endpoints to check for cached JSON payloads prior to running model inference or PostgreSQL lookups. Payloads are cached with a **300-second TTL** using structured keys (`rec:{user_id}:{k}` and `sim:{product_id}:{k}`).
* **Cache Invalidation Strategy (`src/db/load_metadata.py`)**: Implemented pattern-based cache clearing (`rec:*` and `sim:*`) triggered immediately following metadata reloads and catalog updates, ensuring clients never receive stale pricing or missing product attributes.

### Key Components & Architecture

* **Cache Keys**: Standardized naming conventions mapping user and item parameters directly to memory keys.
* **Fail-Safe Operation**: Designed the caching layer to log connection warnings and seamlessly bypass Redis if the service is temporarily unavailable, maintaining high API availability.
* **Invalidation Hook**: Automatically purges relevant cache patterns during ETL catalog updates to maintain data consistency.


## Phase 9: Latency Benchmarking with Locust

This phase adds a Locust load test against the live FastAPI service so throughput and tail latency can be measured under concurrent users, including the effect of Redis caching on repeated `/recommend` and `/similar` traffic.

### What Was Built

* **Locust User (`src/benchmarks/locustfile.py`)**: Defined `RecommenderUser` (`HttpUser`) with a 1–3 second `wait_time` and two `@task` methods: `GET /recommend/0?k=10` and `GET /similar/B0053BCML6?k=10`.
* **Headless runners**: `src/benchmarks/run_benchmark.sh` and `src/benchmarks/run_benchmark.ps1` run Locust without the web UI, writing CSV stats and an HTML report that include Requests/s plus P50, P95, and P99 response times.

### Execution

Start the API first (`python src/api/main.py`), then from the repo root:

```text
powershell -File src/benchmarks/run_benchmark.ps1
```

```text
bash src/benchmarks/run_benchmark.sh
```

Override load with `HOST`, `USERS` (default 20), `SPAWN_RATE` (default 5), and `RUN_TIME` (default `1m`). Direct Locust invocation:

```text
locust -f src/benchmarks/locustfile.py --headless --host http://127.0.0.1:8000 --users 20 --spawn-rate 5 --run-time 1m --csv src/benchmarks/results/locust --html src/benchmarks/results/report.html
```

After the run, `src/benchmarks/results/locust_stats.csv` has **Requests/s** (throughput) and **50% / 95% / 99%** columns (P50 / P95 / P99 latency in milliseconds). `report.html` is the same summary in a browser-readable report.

### Key Components & Architecture

* **Grouped request names**: Locust records `/recommend/{user_id}` and `/similar/{product_id}` so stats stay aggregated rather than exploding into one row per query string.
* **Think time**: `between(1, 3)` seconds models a user pausing between API calls instead of saturating the server with a tight loop.
* **Headless CSV export**: Locust’s `--csv` stats file is the source of truth for comparing cache-cold vs cache-warm latency under the same user count and duration.


## Phase 10: Automated Unit & Integration Testing with Pytest
This phase introduces a robust automated testing and code coverage suite using pytest to guarantee system reliability across core endpoints, database operations, and machine learning inference pipelines.

### What Was Built
Comprehensive Test Suite (tests/): Implemented modular test files covering API routing (tests/test_api.py), database utility logic (tests/test_db.py), and ALS model inference (tests/test_recommender.py).

Isolated Test Fixtures (tests/conftest.py): Configured shared mock fixtures for PostgreSQL connection pools, Redis clients, and fake ALS models to enable fast, offline execution via FastAPI's TestClient.

Automated Coverage Reporting (pytest.ini & .coveragerc): Integrated pytest-cov to enforce code quality metrics, generating terminal coverage tables and browsable HTML reports (htmlcov/).

### Test Execution & Performance
Reliable Execution: Achieved 27 passed tests running locally from the repository root in under a second.

High Coverage: Maintained an overall 85% code coverage rating across the entire service package, with 100% coverage on core API modules and routing layers.

# Phase 11: Containerization with Docker & Multi-Service Compose

## Overview & Architecture
Phase 11 introduces comprehensive containerization to package the entire end-to-end product recommendation system into standardized, isolated, and highly reproducible containers. By utilizing Docker and Docker Compose, we eliminate environment discrepancies ("it works on my machine") and coordinate multi-service architecture seamlessly.

- **Backend Framework**: FastAPI
- **Database**: PostgreSQL 16
- **Cache**: Redis 7
- **Orchestration**: Docker Compose

---

## Architecture Services Breakdown

1. **API Service (`api`)**
   - Built using an optimized Python 3.12 slim base image.
   - Installs required scientific runtime libraries (`libgomp1`, `libpq5`).
   - Installs pinned dependencies cleanly via UTF-8 encoded `requirements.txt`.
   - Mounts local trained models (`artifacts/`) and launches Uvicorn on port `8080`.

2. **Database Service (`db`)**
   - Powered by PostgreSQL 16 Alpine.
   - Configured with secure environment credentials and automated database bootstrapping (`recommender_db`).
   - Utilizes persistent Docker volumes (`postgres_data`) for data durability across lifecycles.

3. **Caching Service (`redis`)**
   - Powered by Redis 7 Alpine mapped to port `6379`.
   - Provides lightning-fast in-memory caching for recommendation queries.

---

## Project File Structure
```text
product-recommender/
├── Dockerfile              # Production image recipe for FastAPI
├── docker-compose.yml      # Multi-service composition & orchestration
├── .dockerignore           # Excludes local caches, venvs, and logs
├── requirements.txt        # UTF-8 encoded Python dependency manifest
├── src/                    # FastAPI application source code
└── artifacts/              # Trained machine learning model weights