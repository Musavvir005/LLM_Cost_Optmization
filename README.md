# Adaptive Dynamic Best-of-N (AD-BoN) Runtime Gating Gateway

An intelligent, multi-tier LLM routing gateway that optimizes quality versus cost at runtime using dynamically classified routing signals, Best-of-N budgeting, prompt and semantic caching, and a lightweight security verification envelope.

---

## Architecture & Secure Request Flow

```
Browser / Client
       |
       | HTTPS (Bearer API_KEY)
       v
Flask API Gateway (/api/query)
       |
       | Authenticated Request (hmac.compare_digest)
       v
Request Classifier (Phi / Probabilities)
       |
       | Routing Signal Envelope (req_id, timestamp, nonce, cluster, target_q)
       v
Signal Integrity & Freshness Verification (HMAC-SHA256 & Nonce Cache)
       |
       v
AD-BoN Gating Optimization Engine
       |
       +------ Fast Tier:     Ollama (phi3 / gemma3)        [$0.00 / 1M]
       |
       +------ Balanced Tier: Groq (compound / qwen-27b)    [$0.59 / 1M]
       |
       +------ Frontier Tier: Google Gemini Pro             [$1.25 / 1M]
       |
       +------ Instant Cache: Semantic Vector Memory        [$0.00 / Instant]
```

---

## Repository Structure

```text
genesis/
├── backend/                  # Core routing services & API endpoints
│   ├── api.py                # FastAPI REST server (port 8000)
│   ├── app.py                # Flask gateway & demo dashboard (port 5000)
│   ├── security_layer.py     # HMAC-SHA256 signing, Bearer auth, replay checks
│   ├── gateway/              # Modular gateway package
│   │   ├── __init__.py
│   │   ├── runtime.py        # Active Phase 4-v4 multimodal routing engine
│   │   └── legacy/           # Archived previous phase implementations (v1-v3)
│   └── benchmark_dashboard.py
├── models/                   # Trained model weights & calibrated profiles
│   ├── saved_router/         # Hugging Face sequence classifier (model.safetensors)
│   ├── kmeans_model.joblib   # Phase 1 KMeans clustering model
│   ├── cluster_centroids.npy # 10 semantic centroid embeddings
│   ├── junior_profile.json   # Fast tier Best-of-N performance profile
│   └── senior_profile.json   # Frontier tier baseline profile
├── data/                     # Datasets, caches & persistent files
│   ├── datasets/             # Cleaned & clustered training CSVs
│   ├── cache/                # Saved semantic cache vectors (semantic_cache.json)
│   └── uploads/              # Uploaded user files (max 10 MB)
├── notebooks/                # Research and training Jupyter notebooks (Phases 1–3)
├── assets/                   # Testing media & evaluation plots
│   ├── samples/              # Sample multimodal files (sample_chart.png, etc.)
│   └── figures/              # Evaluation plots & cluster charts
├── tests/                    # Security & regression test suites (tests/test_security.py)
├── frontend/                 # React 19 + Vite modern dark-mode UI (port 3000)
├── README.md
├── .env.example
└── .gitignore
```

---

## Security Layer

The gateway incorporates a lightweight authentication and secure routing-signal verification layer designed for prototype and research deployments:

1. **API Authentication**:
   - The Flask backend is protected with standard HTTP Bearer token authentication (`Authorization: Bearer <API_KEY>`).
   - Constant-time verification is performed via `hmac.compare_digest()` to eliminate timing attack vectors.
   - Missing or invalid API credentials immediately return standard `HTTP 401 Unauthorized`.
   - API keys are never printed in logs or reflected in response payloads.

2. **Routing Signal Integrity**:
   - Internal routing signals (containing cluster classification, probabilities, target quality $\tau$, and tier selections) are signed using HMAC-SHA256 (`sign_routing_signal`).
   - Signatures are computed over a deterministic canonical JSON representation (`sort_keys=True`, strict compact separators).
   - Any tampering or modification of routing signals in-flight causes verification to fail (`verify_routing_signal`), safely rejecting the signal before gating execution.

3. **Replay Protection**:
   - Every routing signal envelope carries a unique `request_id`, a high-entropy cryptographic `nonce`, and a UTC epoch `timestamp`.
   - The gateway enforces a configurable sliding TTL window (default: 300 seconds / 5 minutes) and tracks observed `(request_id, nonce)` pairs to reject replayed signals.

4. **Environment Configuration**:
   - All secret material is stored exclusively via environment variables (`ADBON_API_KEY` and `ADBON_SIGNAL_SECRET`).
   - Secret keys are never hardcoded in source code, and `.env` is excluded by `.gitignore`.

5. **Deployment Recommendation**:
   - In production environments, all client-to-gateway traffic must be routed over **TLS/HTTPS** to protect credentials and signal payloads in transit.

> **Note**: This security layer is a lightweight student/research prototype built on standard industry primitives (HMAC-SHA256, constant-time digest comparison, bearer authorization). It is **not** a newly invented cryptographic protocol and is designed to integrate cleanly without altering AD-BoN quality/cost optimization dynamics.

---

## Environment Variables

Copy `.env.example` to `.env` (or set environment variables in your shell):

```bash
# Gateway API Authentication Key
ADBON_API_KEY=adbon-sec-key-2026-demo

# HMAC Secret for Routing Signal Integrity
ADBON_SIGNAL_SECRET=adbon-signal-hmac-secret-prototype-2026

# Allowed Replay Time Window (seconds)
ADBON_REPLAY_WINDOW_SECONDS=300

# Server Port
PORT=5000
```

---

## Running the Application

### 1. Start the Flask Secure Gateway
```bash
# Using python directly
python app.py

# Or within the virtual environment
& ".adbon-verify/Scripts/python.exe" app.py
```
The Flask application starts at `http://127.0.0.1:5000/` providing both the REST API and the interactive Web Dashboard.

### 2. Run the Security & Regression Test Suite
A comprehensive 10-test automated suite validates authentication, signal signing, tampering rejection, replay tracking, and existing AD-BoN routing and benchmark mechanics:

```bash
python tests/test_security.py
```

All 10 test scenarios pass:
1. `test_01_valid_api_key_succeeds`: Authenticated request returns 200 with answer and telemetry.
2. `test_02_missing_api_key_unauthorized`: Missing `Authorization` header returns 401.
3. `test_03_wrong_api_key_unauthorized`: Invalid Bearer token returns 401.
4. `test_04_valid_routing_signal_verification`: Valid canonical HMAC-SHA256 verifies successfully.
5. `test_05_modified_routing_signal_fails`: Tampering with cluster or target quality causes signature failure.
6. `test_06_invalid_signature_fails`: Altered signature string is safely rejected.
7. `test_07_expired_timestamp_fails`: Signals older than 5 minutes fail replay validation.
8. `test_08_replay_same_id_nonce_rejected`: Immediate replay of duplicate `(request_id, nonce)` is blocked.
9. `test_09_existing_adbon_routing_passes`: AD-BoN core optimization algorithm operates identically.
10. `test_10_existing_benchmark_suite_passes`: Full 12-query Before/After benchmark executes successfully.

---

## API Reference

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `POST` | `/api/query` | **Protected** | Submit query for AD-BoN routing with HMAC integrity validation |
| `POST` | `/api/benchmark/run` | **Protected** | Execute full 12-query benchmark suite |
| `GET`  | `/api/cache/stats` | **Protected** | Inspect semantic vector cache and prompt-cache statistics |
| `GET`  | `/api/models` | **Protected** | Retrieve model tier specifications and pricing configs |
| `GET`  | `/api/security/status` | **Public** | Status probe reporting active security layers |
| `GET`  | `/` | **Public** | Interactive Web Dashboard |

### Example Request (`POST /api/query`)

```bash
curl -X POST http://127.0.0.1:5000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer adbon-sec-key-2026-demo" \
  -d '{
    "query": "Implement a binary search tree in C++ with insert and search methods.",
    "target_quality": 0.76
  }'
```

### Standardized Response Format

```json
{
  "query": "Implement a binary search tree in C++...",
  "classification": {
    "primary_cluster": "coding",
    "primary_cluster_id": 0,
    "cluster_probability": 0.94,
    "routing_reason": "Confidence guard triggered"
  },
  "routing": {
    "selected_model": "Groq (qwen-27b / compound)",
    "budget": 1,
    "expected_quality": 0.88,
    "target_quality": 0.76
  },
  "cost": {
    "adbon_estimated_cost": 0.000125,
    "baseline_estimated_cost": 0.000625,
    "savings_percent": 80.0
  },
  "cache": {
    "hit": false,
    "tokens_saved": 0
  },
  "security": {
    "authenticated": true,
    "signal_integrity_verified": true,
    "replay_check": true
  },
  "answer": "class BST { ... };"
}
```
