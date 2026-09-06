"""
AD-BoN (Adaptive Dynamic Best-of-N) Runtime Gating Gateway
===========================================================
Phase: Final Integration Demo

IMPORTANT — SYNTHETIC / MOCK NOTICE
-------------------------------------
When run with use_mock_router=True (the default offline mode):
  - The Semantic Router is a DETERMINISTIC RULE-BASED MOCK.
    It uses keyword matching, NOT a trained neural classifier.
  - The Performance Matrix is a HAND-CALIBRATED SYNTHETIC MATRIX.
    It is NOT derived from real benchmark evaluations.
  - Draft generation is SIMULATED. No real LLM calls are made.
  - Reward scoring uses a SYNTHETIC MOCK MODEL.
    Scores are NOT real quality measurements.

To use real components, supply:
  router_path="path/to/saved_router"   -> activates RealRouterWrapper
  profile_paths={...}                  -> loads real benchmark JSON profiles
and plug in a real RewardModel implementing score_completions().

Architecture
------------
  Query
    |
    v
  Semantic Router  Phi(q)           <- SemanticMockRouter or RealRouterWrapper
    |
    v
  Cluster Probability Distribution  <- 10-dim soft probability vector
    |
    v
  Performance Matrix  Psi           <- per-cluster, per-model, per-N quality matrix
    |
    v
  Expected Quality  E[Q|q,model,N] = SUM_k [ Phi_k(q) * Psi(model, k, N) ]
    |
    v
  Cost Estimation                   <- Dynamic Token Costing (KV-cache aware)
    |
    v
  Quality Constraint                <- filter: E[Q] >= target_quality
    |
    v
  Cheapest Valid Configuration      <- argmin cost over satisfying configs
    |
    v
  [Optional] High-Risk Gate         <- override to Senior if high_risk_clusters matched
    |
    v
  Simulated Generation (N drafts)
    |
    v
  Proxy Reward Ranking              <- pick best draft
    |
    v
  Final Response + Diagnostic Report
"""

import os
import sys
import json
import logging
import re
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ADBoNGateway")

# ---------------------------------------------------------------------------
# Guarded imports — allow headless / offline / mock operation
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH_HF = True
except ImportError:
    HAS_TORCH_HF = False
    logger.warning(
        "PyTorch or Hugging Face Transformers not installed. "
        "Running in MOCK ROUTER mode."
    )

try:
    import numpy as np
except ImportError:
    logger.error("NumPy is required. Install with: pip install numpy")
    sys.exit(1)


# ===========================================================================
# CLUSTER TAXONOMY (fixed, 10 semantic clusters)
# ===========================================================================
CLUSTER_NAMES: Dict[int, str] = {
    0: "simple_factual",
    1: "mathematics",
    2: "coding",
    3: "reasoning",
    4: "creative_writing",
    5: "summarization",
    6: "financial_analysis",
    7: "technical_analysis",
    8: "general_knowledge",
    9: "complex_instruction",
}

# Human-readable labels for reporting
CLUSTER_LABELS: Dict[int, str] = {
    0: "Simple Factual",
    1: "Mathematics",
    2: "Coding",
    3: "Reasoning / Logic",
    4: "Creative Writing",
    5: "Summarization",
    6: "Financial Analysis",
    7: "Technical Analysis",
    8: "General Knowledge",
    9: "Complex Instruction",
}

# ---------------------------------------------------------------------------
# Keyword rules for the SemanticMockRouter
# Each entry: (cluster_id, [keyword/phrase list], score_weight)
# Phrases are matched as substrings after lowercasing the query.
# ---------------------------------------------------------------------------
_ROUTING_RULES: List[Tuple[int, List[str], float]] = [
    # Cluster 0 — simple_factual
    (0, ["what is the capital", "who is the president", "when was", "where is",
         "what country", "how many countries", "what year", "who invented",
         "what color", "how tall", "what language", "capital of"], 1.0),

    # Cluster 1 — mathematics
    (1, ["solve", "integral", "derivative", "eigenvalue", "matrix multiplication",
         "proof", "theorem", "equation", "calculate", "probability distribution",
         "linear algebra", "differential equation", "fourier", "laplace",
         "numerical method", "optimization problem", "gradient descent"], 1.0),

    # Cluster 2 — coding
    (2, ["implement", "write a function", "write code", "algorithm", "data structure",
         "python", "java", "c++", "javascript", "red-black tree", "binary tree",
         "linked list", "sort", "search", "graph traversal", "dynamic programming",
         "recursion", "class", "api", "rest api", "debugging", "unit test",
         "design pattern", "refactor", "complexity", "big-o", "deletion",
         "insertion", "balanced tree", "heap", "stack", "queue"], 1.0),

    # Cluster 3 — reasoning
    (3, ["why does", "explain why", "reason", "logical", "deduce", "infer",
         "given that", "therefore", "if then", "causal", "compare and contrast",
         "pros and cons", "evaluate", "argue", "hypothesis", "evidence"], 0.9),

    # Cluster 4 — creative_writing
    (4, ["write a story", "write a poem", "creative", "fiction", "narrative",
         "character", "plot", "dialogue", "screenplay", "short story",
         "write a song", "essay", "blog post", "persuasive"], 1.0),

    # Cluster 5 — summarization
    (5, ["summarize", "summary", "tldr", "briefly explain", "give an overview",
         "key points", "main ideas", "condense", "digest", "abstract",
         "highlights", "outline"], 1.0),

    # Cluster 6 — financial_analysis
    (6, ["financial", "quarterly", "revenue", "earnings", "profit", "loss",
         "balance sheet", "income statement", "cash flow", "risk", "investment",
         "portfolio", "valuation", "market cap", "dividend", "stock", "bond",
         "fiscal", "audit", "forecast", "budget", "expense", "roi", "kpi",
         "ebitda", "gross margin", "core risks", "financial statement"], 1.0),

    # Cluster 7 — technical_analysis
    (7, ["technical analysis", "moving average", "rsi", "macd", "candlestick",
         "support level", "resistance level", "bollinger", "momentum",
         "chart pattern", "trend line", "volume analysis", "oscillator",
         "fibonacci", "market trend", "trading signal"], 1.0),

    # Cluster 8 — general_knowledge
    (8, ["explain", "describe", "tell me about", "overview of",
         "history of", "how does", "definition of", "difference between",
         "types of", "examples of"], 0.6),

    # Cluster 9 — complex_instruction
    (9, ["multi-step", "step by step", "design a system", "architecture",
         "end-to-end", "comprehensive", "full implementation", "detailed plan",
         "from scratch", "production-ready", "scalable", "distributed system",
         "microservice", "pipeline", "workflow"], 1.0),
]


# ===========================================================================
# SemanticMockRouter
# ===========================================================================
class SemanticMockRouter:
    """
    DETERMINISTIC SEMANTIC CLASSIFIER (Req #1).

    Replaces the random MockRouter. Uses keyword matching to assign queries
    to semantic clusters WITHOUT any random numbers, character-sum hashes,
    or random Dirichlet distributions.

    Guarantees:
      - "What is the capital of France?" -> Cluster 0 (simple_factual) ~0.90
      - "Implement a balanced red-black tree with deletion in Python." -> Cluster 2 (coding) ~0.90
      - "Write a summary of the quarterly financial statement highlighting the core risks."
        -> strong distribution toward Cluster 6 (financial_analysis) and Cluster 5 (summarization)
      - Probabilities always sum to exactly 1.0.
      - Validated with assertions on every call.
    """

    def __init__(self, num_clusters: int = 10, primary_weight: float = 0.90):
        if num_clusters != 10:
            raise ValueError(
                "SemanticMockRouter is calibrated for exactly 10 clusters. "
                f"Got num_clusters={num_clusters}."
            )
        self.num_clusters = num_clusters
        self.primary_weight = primary_weight

    def predict_probabilities(self, query: str) -> np.ndarray:
        """
        Classify query into a soft probability distribution over 10 clusters.
        No random numbers used. Validates output distribution with assertions.
        """
        query_lower = query.lower()
        raw_scores = np.zeros(self.num_clusters, dtype=np.float64)

        # Accumulate scores from keyword rules
        for cluster_id, keywords, weight in _ROUTING_RULES:
            for kw in keywords:
                # Word boundaries avoid accidental matches such as "api" in
                # "capital" while still supporting multi-word phrases.
                if re.search(r"(?<!\\w)" + re.escape(kw) + r"(?!\\w)", query_lower):
                    raw_scores[cluster_id] += weight

        if raw_scores.sum() == 0.0:
            # Fallback for queries with no matching keywords
            probs = np.ones(self.num_clusters, dtype=np.float64) / self.num_clusters
        else:
            primary_cluster = int(np.argmax(raw_scores))
            probs = np.zeros(self.num_clusters, dtype=np.float64)
            probs[primary_cluster] = self.primary_weight

            # Distribute remaining mass across other clusters proportional to their scores
            residual = 1.0 - self.primary_weight
            other_scores = raw_scores.copy()
            other_scores[primary_cluster] = 0.0
            other_total = other_scores.sum()

            if other_total > 0.0:
                probs += (other_scores / other_total) * residual
            else:
                n_other = self.num_clusters - 1
                for k in range(self.num_clusters):
                    if k != primary_cluster:
                        probs[k] = residual / n_other

            # Normalize for floating point safety
            probs /= probs.sum()

        # Strict validation checks per Req #1
        assert len(probs) == self.num_clusters, f"Expected {self.num_clusters} probs, got {len(probs)}"
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.isclose(probs.sum(), 1.0), f"Probabilities must sum to 1.0, got {probs.sum()}"

        return probs

    def get_primary_cluster_info(self, query: str) -> Tuple[int, str, float]:
        """Return (cluster_id, cluster_name, probability) for the top cluster."""
        probs = self.predict_probabilities(query)
        primary = int(np.argmax(probs))
        return primary, CLUSTER_NAMES[primary], float(probs[primary])

    def get_routing_reason(self, query: str) -> str:
        """Return a human-readable explanation of which keywords triggered routing."""
        query_lower = query.lower()
        primary_id, primary_name, _ = self.get_primary_cluster_info(query)
        matched = []
        for cluster_id, keywords, _ in _ROUTING_RULES:
            if cluster_id == primary_id:
                for kw in keywords:
                    if re.search(r"(?<!\\w)" + re.escape(kw) + r"(?!\\w)", query_lower):
                        matched.append(f'"{kw}"')
        if matched:
            return f"Detected {primary_name} terminology: {', '.join(matched)}"
        return f"Assigned to {primary_name} by semantic rule evaluation."


# Backward compatibility alias
MockRouter = SemanticMockRouter


# ===========================================================================
# RealRouterWrapper — Real HuggingFace Sequence Classification
# ===========================================================================
class RealRouterWrapper:
    """
    Wrapper for the Phase 3 trained sequence classification router.
    Preserves real neural model inference with AutoTokenizer and
    AutoModelForSequenceClassification.
    """

    def __init__(self, model_path: str, num_clusters: int = 10):
        if not HAS_TORCH_HF:
            raise RuntimeError(
                "PyTorch and Hugging Face Transformers are required to load "
                f"the real router from '{model_path}'."
            )
        self.num_clusters = num_clusters
        self.model_path = model_path

        logger.info(f"Loading real router from '{model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Real router loaded successfully on device: {self.device}")

    def predict_probabilities(self, query: str) -> np.ndarray:
        """
        Run forward pass of fine-tuned sequence classifier and return softmax probabilities.
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        if len(probs) != self.num_clusters:
            raise ValueError(
                f"Model output {len(probs)} classes, expected {self.num_clusters}."
            )
        return probs.astype(np.float64)


# ===========================================================================
# SyntheticMockRewardModel
# ===========================================================================
class SyntheticMockRewardModel:
    """
    This is NOT a real reward model.
    It produces synthetic deterministic scores only for testing.
    Scores produced by this class are NOT real quality measurements.
    They exist solely to exercise the Best-of-N re-ranking step of the
    AD-BoN pipeline in an offline, dependency-free setting.
    """

    def score_completions(self, completions: List[str]) -> List[float]:
        """
        Score candidate completions deterministically based on text and index.
        """
        scores = []
        for idx, c in enumerate(completions):
            content_hash = sum(ord(ch) for ch in c) % 997
            seed = (content_hash * 31 + idx * 17) % (2**31)
            rng = np.random.default_rng(seed)
            score = float(np.clip(rng.normal(0.65, 0.15), 0.10, 0.99))
            scores.append(score)
        return scores


# Backward compatibility alias
MockRewardModel = SyntheticMockRewardModel


# ===========================================================================
# SYNTHETIC PERFORMANCE MATRIX (Psi)
# ===========================================================================
def _build_synthetic_performance_matrix(num_clusters: int = 10) -> Dict[str, Any]:
    """
    Build a cluster-specific synthetic performance matrix for all three model tiers.

    Uses the calibrated base qualities per cluster:
      - simple_factual (0): Junior base = 0.72 -> benefits from inexpensive Best-of-N
      - coding (2): Junior base = 0.35 -> Bo3 (~0.58) and Bo5 (~0.71) fail 0.76; Medium Bo1 (0.80) clears
      - financial_analysis (6): Junior base = 0.40 -> Bo5 (~0.73) fails 0.76; Medium Bo1 (0.83) clears

    Diminishing returns formula for Junior and Medium:
      bo3 = base + (1.0 - base) * 0.35
      bo5 = base + (1.0 - base) * 0.55
      Bo1 < Bo3 < Bo5 holds strictly.

    Senior has Bo1 only.
    """
    # Junior Bo1 base quality per cluster
    junior_base = {
        0: 0.72,  # simple_factual
        1: 0.42,  # mathematics
        2: 0.35,  # coding: Junior fails even at Bo5 (0.35 + 0.65*0.55 = 0.7075 < 0.76)
        3: 0.45,  # reasoning
        4: 0.68,  # creative_writing
        5: 0.70,  # summarization
        6: 0.40,  # financial_analysis: Junior fails even at Bo5 (0.40 + 0.60*0.55 = 0.73 < 0.76)
        7: 0.42,  # technical_analysis
        8: 0.68,  # general_knowledge
        9: 0.38,  # complex_instruction
    }

    # Medium Bo1 base quality per cluster
    medium_base = {
        0: 0.84,
        1: 0.78,
        2: 0.80,
        3: 0.81,
        4: 0.82,
        5: 0.84,
        6: 0.83,
        7: 0.81,
        8: 0.83,
        9: 0.80,
    }

    # Senior Bo1 base quality per cluster
    senior_base = {
        0: 0.94,
        1: 0.93,
        2: 0.95,
        3: 0.95,
        4: 0.93,
        5: 0.94,
        6: 0.95,
        7: 0.94,
        8: 0.94,
        9: 0.95,
    }

    matrix: Dict[str, Dict[str, Dict[str, float]]] = {
        "junior":  {"performance_matrix": {}},
        "medium":  {"performance_matrix": {}},
        "senior":  {"performance_matrix": {}},
    }

    for k in range(num_clusters):
        # Junior
        j_b1 = junior_base[k]
        j_b3 = round(j_b1 + (1.0 - j_b1) * 0.35, 4)
        j_b5 = round(j_b1 + (1.0 - j_b1) * 0.55, 4)
        matrix["junior"]["performance_matrix"][str(k)] = {
            "1": round(j_b1, 4),
            "3": j_b3,
            "5": j_b5,
        }

        # Medium
        m_b1 = medium_base[k]
        m_b3 = round(m_b1 + (1.0 - m_b1) * 0.35, 4)
        m_b5 = round(m_b1 + (1.0 - m_b1) * 0.55, 4)
        matrix["medium"]["performance_matrix"][str(k)] = {
            "1": round(m_b1, 4),
            "3": m_b3,
            "5": m_b5,
        }

        # Senior (Bo1 only)
        matrix["senior"]["performance_matrix"][str(k)] = {
            "1": round(senior_base[k], 4),
        }

    for key in matrix:
        matrix[key]["model"] = key
        matrix[key]["source"] = "synthetic_cluster_dependent"

    return matrix


# ===========================================================================
# ADBoNGateway — Main Gateway Class
# ===========================================================================
class ADBoNGateway:
    """
    AD-BoN Active Gating Runtime Gateway.

    Routes incoming queries to the cheapest LLM configuration whose
    expected quality meets the specified target_quality threshold.

    Expected quality formula:
        E[Q | q, model, N] = SUM_k [ Phi_k(q) * Psi(model, k, N) ]
    """

    def __init__(
        self,
        router_path: Optional[str] = None,
        profile_paths: Optional[Dict[str, str]] = None,
        num_clusters: int = 10,
        use_mock_router: bool = False,
        enable_high_risk_gate: bool = False,
        high_risk_clusters: Optional[List[int]] = None,
        reward_model: Any = None,
    ):
        self.num_clusters = num_clusters
        self.enable_high_risk_gate = enable_high_risk_gate
        self.high_risk_clusters = high_risk_clusters if high_risk_clusters is not None else [2, 3, 9]

        # 1. Semantic Router (Phi)
        if use_mock_router or not HAS_TORCH_HF or not router_path or not os.path.exists(router_path):
            if router_path and not os.path.exists(router_path):
                logger.warning(f"Router path '{router_path}' not found. Using SemanticMockRouter.")
            logger.info("[MOCK MODE] Using SemanticMockRouter — deterministic keyword-based classifier.")
            self.router = SemanticMockRouter(num_clusters=num_clusters)
            self.is_mock_router = True
        else:
            self.router = RealRouterWrapper(router_path, num_clusters=num_clusters)
            self.is_mock_router = False

        # 2. Model Specifications & Pricing
        self.model_configs: Dict[str, Any] = {
            "junior": {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "price_input_m":  0.15,
                "price_output_m": 0.60,
                "budgets": [1, 3, 5],
                "avg_output_len": 180,
            },
            "medium": {
                "name": "meta-llama/Meta-Llama-3-70B-Instruct",
                "price_input_m":  0.70,
                # Kept above three Junior drafts so a simple query that
                # needs Bo3 can still select the economical Junior tier.
                "price_output_m": 1.60,
                "budgets": [1, 3, 5],
                "avg_output_len": 220,
            },
            "senior": {
                "name": "gpt-4o",
                "price_input_m":  5.00,
                "price_output_m": 15.00,
                "budgets": [1],
                "avg_output_len": 250,
            },
        }

        # 3. Performance Profiles (Psi Matrix)
        synthetic_all = _build_synthetic_performance_matrix(num_clusters)
        self.profiles: Dict[str, Any] = {}

        for model_key in self.model_configs:
            path = (profile_paths.get(model_key) if profile_paths else None)
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    loaded = json.load(f)
                self._validate_profile(loaded, model_key)
                self.profiles[model_key] = loaded
                logger.info(f"Loaded real performance profile for '{model_key}' from {path}")
            else:
                self.profiles[model_key] = synthetic_all[model_key]

        # 4. Reward Model
        if reward_model is not None:
            self.reward_model = reward_model
        else:
            self.reward_model = SyntheticMockRewardModel()

    def _validate_phi(self, phi: np.ndarray) -> None:
        """Validate probability distribution vector."""
        if len(phi) != self.num_clusters:
            raise ValueError(f"Router returned {len(phi)} probabilities, expected {self.num_clusters}.")
        if not np.all(np.isfinite(phi)):
            raise ValueError("Router returned non-finite probability values.")
        if np.any(phi < 0):
            raise ValueError("Router returned negative probability values.")
        if not np.isclose(phi.sum(), 1.0):
            raise ValueError(f"Router probability vector does not sum to 1.0 (got {phi.sum():.8f}).")

    def _validate_profile(self, profile: Dict[str, Any], model_key: str) -> None:
        """Ensure performance profile has all clusters and budgets."""
        required_budgets = {
            "junior": ["1", "3", "5"],
            "medium": ["1", "3", "5"],
            "senior": ["1"],
        }
        matrix = profile.get("performance_matrix", {})
        budgets = required_budgets.get(model_key, ["1"])
        for k in range(self.num_clusters):
            sk = str(k)
            if sk not in matrix:
                raise ValueError(f"Profile for '{model_key}' is missing cluster {k}.")
            for b in budgets:
                if b not in matrix[sk]:
                    raise ValueError(f"Profile for '{model_key}' cluster {k} is missing budget N={b}.")

    def _estimate_input_tokens(self, prompt: str) -> int:
        """
        Estimate the number of input tokens.
        If a tokenizer is available, use the tokenizer.
        Otherwise: return max(1, len(prompt) // 4)
        """
        if not self.is_mock_router and hasattr(self.router, "tokenizer") and self.router.tokenizer is not None:
            try:
                return len(self.router.tokenizer.encode(prompt))
            except Exception:
                pass
        return max(1, len(prompt) // 4)

    def _calculate_token_cost(self, prompt: str, model_key: str, budget_n: int) -> float:
        """
        Dynamic token cost estimation with KV-cache awareness.
        """
        config = self.model_configs[model_key]
        p_in   = config["price_input_m"]
        p_out  = config["price_output_m"]
        l_out  = config["avg_output_len"]

        l_in = self._estimate_input_tokens(prompt)

        # Single prompt ingestion + N parallel generation outputs
        input_cost  = (l_in / 1_000_000.0) * p_in
        output_cost = (l_out * budget_n / 1_000_000.0) * p_out
        return float(input_cost + output_cost)

    def _compute_expected_quality(self, phi: np.ndarray, model_key: str, budget_n: int) -> float:
        """
        Compute expected quality: E[Q | q, model, N] = SUM_k [ Phi_k(q) * Psi(model, k, N) ]
        """
        matrix = self.profiles[model_key]["performance_matrix"]
        b_str  = str(budget_n)
        psi_vector = np.array([matrix[str(k)][b_str] for k in range(self.num_clusters)], dtype=np.float64)
        return float(np.dot(phi, psi_vector))

    def route_query(self, query: str, target_quality: float = 0.76) -> Dict[str, Any]:
        """
        Active Dynamic Best-of-N routing decision.
        """
        if not (0 < target_quality <= 1.0):
            raise ValueError(f"target_quality must be in (0, 1]. Got {target_quality}.")

        # 1. Predict cluster distribution
        phi = self.router.predict_probabilities(query)
        self._validate_phi(phi)

        primary_cluster = int(np.argmax(phi))
        primary_prob    = float(phi[primary_cluster])
        primary_name    = CLUSTER_NAMES[primary_cluster]
        primary_label   = CLUSTER_LABELS[primary_cluster]

        if self.is_mock_router and hasattr(self.router, "get_routing_reason"):
            routing_reason = self.router.get_routing_reason(query)
        else:
            routing_reason = f"Predicted primary cluster: {primary_label}"

        # 2. Enumerate all candidates
        candidates = []
        for model_key, config in self.model_configs.items():
            for budget_n in config["budgets"]:
                eq   = self._compute_expected_quality(phi, model_key, budget_n)
                cost = self._calculate_token_cost(query, model_key, budget_n)
                candidates.append({
                    "model":            model_key,
                    "budget":           budget_n,
                    "expected_quality": eq,
                    "cost":             cost,
                    "name":             config["name"],
                })

        # 3. Cost-aware quality optimization
        satisfying = [c for c in candidates if c["expected_quality"] >= target_quality]

        if satisfying:
            selected_route = min(satisfying, key=lambda x: x["cost"])
            fallback_used  = False
        else:
            selected_route = next(c for c in candidates if c["model"] == "senior")
            fallback_used  = True

        # 4. Optional High-risk override gate
        high_risk_override = False
        if self.enable_high_risk_gate and primary_cluster in self.high_risk_clusters:
            high_risk_override = True
            selected_route = next(c for c in candidates if c["model"] == "senior")

        # 5. Cost comparison vs Senior
        senior_cost = next(c for c in candidates if c["model"] == "senior")["cost"]
        cost_saving_pct = max(
            0.0,
            ((senior_cost - selected_route["cost"]) / senior_cost) * 100.0,
        )

        # 6. Simulated generation & reward re-ranking
        final_response, raw_drafts, draft_scores = self._execute_generation_and_scoring(
            query, selected_route["model"], selected_route["budget"]
        )

        return {
            "query":                       query,
            "target_quality":              target_quality,
            "predicted_distribution":      phi,
            "primary_cluster":             primary_cluster,
            "primary_cluster_probability": primary_prob,
            "primary_cluster_prob":        primary_prob,
            "primary_cluster_name":        primary_name,
            "primary_cluster_label":       primary_label,
            "routing_reason":              routing_reason,
            "all_candidates":              candidates,
            "selected_route":              selected_route,
            "fallback_used":               fallback_used,
            "high_risk_override":          high_risk_override,
            "cost_saving_percentage":      cost_saving_pct,
            "final_response":              final_response,
            "raw_drafts":                  raw_drafts,
            "draft_scores":                draft_scores,
            "senior_cost_baseline":        senior_cost,
            "is_mock_mode":                self.is_mock_router,
        }

    def _execute_generation_and_scoring(
        self,
        query: str,
        model_key: str,
        budget_n: int,
    ) -> Tuple[str, List[str], List[float]]:
        """
        Distinct synthetic drafts per candidate index and model tier.
        """
        display_names = {
            "junior": "Phi-3-mini",
            "medium": "Llama-3-70B",
            "senior": "GPT-4o",
        }
        display = display_names.get(model_key, model_key)
        q_short = query[:50] + ("..." if len(query) > 50 else "")

        templates = [
            f"[SIMULATED DRAFT — {display}] Direct answer: Key findings for '{q_short}' are addressed directly with concise clarity.",
            f"[SIMULATED DRAFT — {display}] Structured walkthrough: Outlining a methodical step-by-step breakdown for '{q_short}'.",
            f"[SIMULATED DRAFT — {display}] Deep analytical perspective: Examining foundational constraints and edge cases for '{q_short}'.",
            f"[SIMULATED DRAFT — {display}] Practical synthesis: Highlighting core functional requirements and practical guidance for '{q_short}'.",
            f"[SIMULATED DRAFT — {display}] Exhaustive review: Comprehensive, rigorous treatment addressing all facets of '{q_short}'.",
        ]

        raw_drafts = []
        for i in range(budget_n):
            raw_drafts.append(templates[i % len(templates)])

        scores = self.reward_model.score_completions(raw_drafts)
        best_idx = int(np.argmax(scores))
        return raw_drafts[best_idx], raw_drafts, scores


# ===========================================================================
# Diagnostic Report Printer
# ===========================================================================
def print_diagnostic_report(report: Dict[str, Any]) -> None:
    """
    Print a structured diagnostic report using report["target_quality"].
    """
    selected = report["selected_route"]
    tq       = report["target_quality"]
    W        = 78

    print("=" * W)
    print("AD-BoN RUNTIME GATING GATEWAY REPORT")
    if report.get("is_mock_mode"):
        print("  [ SYNTHETIC MOCK MODE — results are simulated ]")
    print("=" * W)

    print(f"\nUser Query:")
    print(f"{report['query']}")

    pc_id   = report["primary_cluster"]
    pc_name = CLUSTER_NAMES[pc_id]
    pc_prob = report["primary_cluster_probability"]

    print(f"\nPrimary Semantic Cluster:")
    print(f"  {pc_name} (Cluster {pc_id}): {pc_prob*100:.1f}%")
    print(f"  Reason: {report['routing_reason']}")

    print("\nPredicted Cluster Distribution  Phi(q):")
    phi = report["predicted_distribution"]
    for k, p in enumerate(phi):
        if p > 0.02:
            bar = "#" * int(p * 25)
            label = CLUSTER_NAMES.get(k, f"cluster_{k}")
            marker = " <-- PRIMARY" if k == pc_id else ""
            print(f"  [{k}] {label:<22} {p*100:5.1f}%  {bar}{marker}")

    print(f"\nTarget Quality Threshold: {tq:.4f}")
    print("\nModel Configurations:")
    print(f"  {'Model':<8} {'N':>3} {'E[Q]':>8} {'Estimated Cost':>16} {'Valid?':>10}")
    print("  " + "-" * 50)
    for c in report["all_candidates"]:
        is_valid = c["expected_quality"] >= tq
        valid_str = "VALID" if is_valid else "INVALID"
        sel_marker = " <-- SELECTED" if c is selected else ""
        print(
            f"  {c['model']:<8} {c['budget']:>3} {c['expected_quality']:>8.4f} "
            f"${c['cost']:>14.8f} {valid_str:>10}{sel_marker}"
        )

    print("\nDECISION:")
    if report.get("high_risk_override"):
        print(f"  [HIGH-RISK GATE ACTIVE] Cluster '{pc_name}' forced Senior.")
    elif report["fallback_used"]:
        print("  [FALLBACK] No candidate met target quality. Defaulting to Senior.")
    else:
        print("  [OPTIMIZER] Cheapest valid configuration selected.")

    print(f"  Selected Model        : {selected['model'].upper()} ({selected['name']})")
    print(f"  Selected Budget       : N = {selected['budget']}")
    print(f"  Expected Quality      : {selected['expected_quality']:.4f}  (threshold: {tq:.4f})")
    print(f"  Estimated Query Cost  : ${selected['cost']:.8f}")
    print(f"  Senior Baseline       : ${report['senior_cost_baseline']:.8f}")
    if not report["fallback_used"] and not report.get("high_risk_override"):
        print(f"  Estimated Cost Saving : {report['cost_saving_percentage']:.2f}% vs. always-Senior")

    print("\nGENERATED SAMPLES & SYNTHETIC REWARD SCORING:")
    print("  (Scores are SIMULATED — not real quality measurements)")
    for idx, (draft, score) in enumerate(zip(report["raw_drafts"], report["draft_scores"])):
        chosen = "*** CHOSEN ***" if draft == report["final_response"] else ""
        print(f"  [{idx+1}] Score: {score:.4f} {chosen}")
        print(f"       {draft[:70]}...")
    print("=" * W + "\n")


# ===========================================================================
# Assertion-Based Regression Tests (Req #12)
# ===========================================================================
def run_regression_tests():
    """
    Automated regression tests verifying deterministic routing and valid distributions.
    """
    gateway = ADBoNGateway(use_mock_router=True)

    easy = gateway.route_query(
        "What is the capital of France?",
        target_quality=0.76
    )

    coding = gateway.route_query(
        "Implement a balanced red-black tree with deletion in Python.",
        target_quality=0.76
    )

    finance = gateway.route_query(
        "Write a summary of the quarterly financial statement highlighting the core risks.",
        target_quality=0.76
    )

    # Primary cluster checks
    assert easy["primary_cluster"] == 0, f"Expected 0, got {easy['primary_cluster']}"
    assert coding["primary_cluster"] == 2, f"Expected 2, got {coding['primary_cluster']}"
    assert finance["primary_cluster"] in [5, 6], f"Expected 5 or 6, got {finance['primary_cluster']}"

    # Probability sum checks
    assert np.isclose(easy["predicted_distribution"].sum(), 1.0), "easy probs do not sum to 1.0"
    assert np.isclose(coding["predicted_distribution"].sum(), 1.0), "coding probs do not sum to 1.0"
    assert np.isclose(finance["predicted_distribution"].sum(), 1.0), "finance probs do not sum to 1.0"

    # Routing differentiation checks (not all Junior Bo5!).  Do not assert
    # an exact easy-query tier: costs and supplied profiles may be calibrated.
    assert not (coding["selected_route"]["model"] == "junior" and coding["selected_route"]["budget"] == 5)
    assert not (finance["selected_route"]["model"] == "junior" and finance["selected_route"]["budget"] == 5)
    assert coding["selected_route"]["model"] == "medium"
    assert finance["selected_route"]["model"] == "medium"

    print("ALL AD-BoN REGRESSION TESTS PASSED")


# ===========================================================================
# Entry Point
# ===========================================================================
if __name__ == "__main__":
    gateway = ADBoNGateway(
        use_mock_router=True,
        enable_high_risk_gate=False,
    )

    test_queries = [
        "What is the capital of France?",
        "Implement a balanced red-black tree with deletion in Python.",
        "Write a summary of the quarterly financial statement highlighting the core risks.",
    ]

    TARGET_QUALITY = 0.76

    for query in test_queries:
        result = gateway.route_query(query, target_quality=TARGET_QUALITY)
        print_diagnostic_report(result)

    # Run automated regression tests
    run_regression_tests()
