"""
Phase 2: Offline LLM Profiling — Model "Report Cards" (Psi matrices)
=====================================================================
AD-BoN routing framework. Builds on:
  - Phase 0: dataset_clean.csv
  - Phase 1: dataset_clustered.csv, kmeans_model.joblib

This script profiles a Junior and a Senior model's Best-of-N performance
across semantic clusters, producing Psi(M_i) in R^{K x N} matrices that
the runtime router will use for gating decisions.
"""

from __future__ import annotations

import os

# Set your API keys in your environment, or paste them here directly for local testing
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "AIzaSy...")  # Your Google AI Studio Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "gsk_...")      # Your Groq Developer Key

import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase2")

# ============================== CONFIG ==============================
INPUT_PATH = "dataset_clustered.csv"

JUNIOR_PROFILE_OUT = "junior_profile.json"
SENIOR_PROFILE_OUT = "senior_profile.json"
RAW_SCORES_OUT = "profiling_raw_scores.csv"

PROMPT_COL = "Prompt"
CLUSTER_COL = "cluster_id"

SAMPLES_PER_CLUSTER = 20          # M = representative prompts sampled per cluster
BUDGETS = [1, 3, 5]               # n values to evaluate Best-of-N at
JUNIOR_TEMPERATURE = 0.7

SIMULATION_MODE = True            # True = mock inference + mock rewards, fully offline
RANDOM_STATE = 42

MODEL_CONFIG = {
    "junior": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "scale_with_n": True,
    },
    "senior": {
        "name": "gpt-4o",
        "scale_with_n": False,   # senior always runs n=1, per spec
    },
}

REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
# ======================================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


@dataclass
class Draft:
    prompt_id: str
    cluster_id: int
    model_key: str      # "junior" or "senior"
    n_budget: int        # which BoN budget this draft belongs to
    draft_index: int     # index within that budget's parallel drafts
    text: str
    latency_sec: float
    reward_score: Optional[float] = None


# ----------------------------------------------------------------------
# Step 1: Load assets
# ----------------------------------------------------------------------
def load_clustered_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        log.error(f"Input file not found: {p.resolve()}")
        sys.exit(1)

    df = pd.read_csv(p)
    required = {PROMPT_COL, CLUSTER_COL}
    missing = required - set(df.columns)
    if missing:
        log.error(f"Missing required column(s): {missing}. Found: {df.columns.tolist()}")
        sys.exit(1)

    log.info(f"Loaded {len(df)} rows across {df[CLUSTER_COL].nunique()} clusters.")
    return df


# ----------------------------------------------------------------------
# Step 2: Subsample representative prompts per cluster
# ----------------------------------------------------------------------
def subsample_per_cluster(df: pd.DataFrame, m: int) -> pd.DataFrame:
    """
    Sample up to m prompts per cluster_id. If a cluster has fewer than m
    rows, take all of them.
    """
    sampled_parts = []
    for cluster_id, group in df.groupby(CLUSTER_COL):
        if len(group) < m:
            log.warning(
                f"Cluster {cluster_id} has only {len(group)} prompts (< {m}); "
                f"using all of them."
            )
            sampled_parts.append(group)
        else:
            sampled_parts.append(group.sample(n=m, random_state=RANDOM_STATE))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    log.info(f"Subsampled {len(sampled)} prompts total for profiling.")
    return sampled


# ----------------------------------------------------------------------
# Step 3: Inference (simulation + real placeholders)
# ----------------------------------------------------------------------
class InferenceEngine:
    """
    Generates candidate drafts for a given prompt/model/budget.
    In simulation mode, produces mock text + latency.
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode

    def generate_drafts(
        self, prompt: str, model_key: str, n_budget: int, temperature: float
    ) -> List[Draft]:
        if self.simulation_mode:
            return self._simulate_drafts(prompt, model_key, n_budget)
        return self._real_drafts(prompt, model_key, n_budget, temperature)

    def _simulate_drafts(self, prompt: str, model_key: str, n_budget: int) -> List[Draft]:
        drafts = []
        n_to_generate = n_budget if MODEL_CONFIG[model_key]["scale_with_n"] else 1
        for i in range(n_to_generate):
            base_latency = 1.8 if model_key == "senior" else 0.4
            latency = max(0.05, float(np.random.normal(base_latency, base_latency * 0.15)))
            drafts.append(
                Draft(
                    prompt_id="",
                    cluster_id=-1,
                    model_key=model_key,
                    n_budget=n_budget,
                    draft_index=i,
                    text=f"[SIMULATED {model_key} draft {i} for budget n={n_budget}]",
                    latency_sec=latency,
                )
            )
        return drafts

    def _real_drafts(
        self, prompt: str, model_key: str, n_budget: int, temperature: float
    ) -> List[Draft]:
        raise NotImplementedError("Real inference placeholder")


# ----------------------------------------------------------------------
# Step 4: Reward scoring
# ----------------------------------------------------------------------
class RewardModel:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode

    def score(self, draft: Draft) -> float:
        if self.simulation_mode:
            return self._simulate_score(draft)
        return self._real_score(draft)

    def _simulate_score(self, draft: Draft) -> float:
        if draft.model_key == "senior":
            mean = 0.80
            std = 0.05
        else:
            k = 0.10
            mean = 0.55 + k * math.log(draft.n_budget)
            std = 0.10

        raw_score = np.random.normal(mean, std)
        return float(np.clip(raw_score, 0.0, 1.0))

    def _real_score(self, draft: Draft) -> float:
        raise NotImplementedError("Real reward scoring placeholder")


# ----------------------------------------------------------------------
# Step 5: Best-of-N re-ranking
# ----------------------------------------------------------------------
def best_of_n_score(drafts: List[Draft]) -> float:
    scored = [d.reward_score for d in drafts if d.reward_score is not None]
    if not scored:
        raise ValueError("No scored drafts provided to best_of_n_score().")
    return max(scored)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def profile_model(
    model_key: str,
    sampled_df: pd.DataFrame,
    engine: InferenceEngine,
    reward_model: RewardModel,
) -> pd.DataFrame:
    from tqdm import tqdm

    results = []
    scale_with_n = MODEL_CONFIG[model_key]["scale_with_n"]
    budgets_to_run = BUDGETS if scale_with_n else [1]

    log.info(f"Profiling model='{model_key}' across {len(sampled_df)} prompts, budgets={budgets_to_run}")

    for row in tqdm(sampled_df.itertuples(index=False), total=len(sampled_df), desc=f"Profiling {model_key}"):
        prompt_id = str(getattr(row, "ID", None) or hash(getattr(row, PROMPT_COL)))
        prompt_text = getattr(row, PROMPT_COL)
        cluster_id = getattr(row, CLUSTER_COL)

        for n_budget in budgets_to_run:
            drafts = engine.generate_drafts(
                prompt=prompt_text,
                model_key=model_key,
                n_budget=n_budget,
                temperature=JUNIOR_TEMPERATURE,
            )
            for d in drafts:
                d.prompt_id = prompt_id
                d.cluster_id = cluster_id
                d.reward_score = reward_model.score(d)

            bon = best_of_n_score(drafts)
            mean_latency = float(np.mean([d.latency_sec for d in drafts]))

            report_budgets = BUDGETS if not scale_with_n else [n_budget]
            for rb in report_budgets:
                results.append({
                    "prompt_id": prompt_id,
                    "cluster_id": cluster_id,
                    "model_key": model_key,
                    "n_budget": rb,
                    "bon_score": bon,
                    "mean_latency_sec": mean_latency,
                })

    return pd.DataFrame(results)


def build_psi_matrix(results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    psi: Dict[str, Dict[str, float]] = {}
    grouped = results_df.groupby(["cluster_id", "n_budget"])["bon_score"].mean()

    for cluster_id in sorted(results_df["cluster_id"].unique()):
        psi[str(cluster_id)] = {}
        for n_budget in BUDGETS:
            key = f"n={n_budget}"
            if (cluster_id, n_budget) in grouped.index:
                psi[str(cluster_id)][key] = round(float(grouped.loc[(cluster_id, n_budget)]), 4)
            else:
                psi[str(cluster_id)][key] = None
    return psi


def save_json(data: dict, path: str) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Saved profile to {path}")
    except OSError as e:
        log.error(f"Failed to write {path}: {e}")
        raise


def log_junior_progression(psi_junior: Dict[str, Dict[str, float]]) -> None:
    log.info("Junior model average score progression (across all clusters):")
    for n_budget in BUDGETS:
        key = f"n={n_budget}"
        vals = [v[key] for v in psi_junior.values() if v.get(key) is not None]
        if vals:
            log.info(f"  n={n_budget}: mean score = {np.mean(vals):.4f} (over {len(vals)} clusters)")


def main() -> None:
    start_time = time.time()

    df = load_clustered_dataset(INPUT_PATH)
    sampled_df = subsample_per_cluster(df, SAMPLES_PER_CLUSTER)

    engine = InferenceEngine(simulation_mode=SIMULATION_MODE)
    reward_model = RewardModel(simulation_mode=SIMULATION_MODE)

    all_results = []
    for model_key in MODEL_CONFIG:
        model_results = profile_model(model_key, sampled_df, engine, reward_model)
        all_results.append(model_results)

    results_df = pd.concat(all_results, ignore_index=True)

    try:
        results_df.to_csv(RAW_SCORES_OUT, index=False)
        log.info(f"Saved raw profiling scores to {RAW_SCORES_OUT}")
    except OSError as e:
        log.error(f"Failed to write {RAW_SCORES_OUT}: {e}")

    junior_results = results_df[results_df["model_key"] == "junior"]
    senior_results = results_df[results_df["model_key"] == "senior"]

    psi_junior = build_psi_matrix(junior_results)
    psi_senior = build_psi_matrix(senior_results)

    save_json(psi_junior, JUNIOR_PROFILE_OUT)
    save_json(psi_senior, SENIOR_PROFILE_OUT)

    log_junior_progression(psi_junior)

    elapsed = time.time() - start_time
    log.info(f"Phase 2 complete in {elapsed:.1f}s. "
              f"({'SIMULATION' if SIMULATION_MODE else 'REAL'} mode)")


if __name__ == "__main__":
    main()
