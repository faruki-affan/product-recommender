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

