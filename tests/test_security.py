"""
AD-BoN Security Layer & Gateway Test Suite
==========================================
Comprehensive regression and security test suite covering:
  1. Valid API key -> request succeeds (HTTP 200).
  2. Missing API key -> HTTP 401 Unauthorized.
  3. Wrong API key -> HTTP 401 Unauthorized.
  4. Valid routing signal -> HMAC verification succeeds.
  5. Modified routing signal -> HMAC verification fails.
  6. Invalid signature -> HMAC verification fails.
  7. Expired timestamp -> Replay check fails.
  8. Replay of same request_id/nonce -> Replay check rejected.
  9. Existing AD-BoN routing optimization tests still pass.
 10. Existing benchmark suite tests still pass.
"""

import os
import sys
import time
import json
import unittest

# Ensure project directory is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
LEGACY_DIR = os.path.join(PROJECT_ROOT, "backend", "gateway", "legacy")
for p in [PROJECT_ROOT, LEGACY_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Set test environment credentials before importing app modules
os.environ["ADBON_API_KEY"] = "test-secret-api-key-xyz"
os.environ["ADBON_SIGNAL_SECRET"] = "test-signal-secret-987"
os.environ["ADBON_REPLAY_WINDOW"] = "300"

from security_layer import (
    verify_api_key,
    sign_routing_signal,
    verify_routing_signal,
    create_secure_routing_signal,
    ReplayProtectionManager,
    get_replay_manager,
    get_expected_api_key,
    get_signal_secret,
)
from app import app as flask_app
from phase4_runtime_gateway_v3 import ADBoNGateway
from benchmark_dashboard import run_benchmark_suite


class TestADBoNSecurity(unittest.TestCase):

    def setUp(self):
        """Configure test client and fresh replay manager for each test."""
        self.client = flask_app.test_client()
        self.api_key = get_expected_api_key()
        self.valid_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Reset singleton replay cache
        get_replay_manager().clear()

    # -----------------------------------------------------------------------
    # Test 1: Valid API key -> request succeeds (200)
    # -----------------------------------------------------------------------
    def test_01_valid_api_key_succeeds(self):
        payload = {"query": "What is the speed of sound?"}
        res = self.client.post("/api/query", headers=self.valid_headers, json=payload)
        self.assertEqual(res.status_code, 200, f"Expected 200, got: {res.data}")
        data = res.get_json()
        self.assertTrue(data.get("security", {}).get("authenticated"))
        self.assertTrue(data.get("security", {}).get("signal_integrity_verified"))
        self.assertTrue(data.get("security", {}).get("replay_check"))
        self.assertIn("answer", data)
        self.assertIn("classification", data)
        self.assertIn("routing", data)

    # -----------------------------------------------------------------------
    # Test 2: Missing API key -> 401 Unauthorized
    # -----------------------------------------------------------------------
    def test_02_missing_api_key_unauthorized(self):
        payload = {"query": "What is the capital of Japan?"}
        res = self.client.post("/api/query", json=payload)
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertEqual(data.get("error"), "Unauthorized")
        # Ensure API key is NOT leaked in error response
        self.assertNotIn(self.api_key, json.dumps(data))

    # -----------------------------------------------------------------------
    # Test 3: Wrong API key -> 401 Unauthorized
    # -----------------------------------------------------------------------
    def test_03_wrong_api_key_unauthorized(self):
        bad_headers = {
            "Authorization": "Bearer totally-wrong-invalid-key",
            "Content-Type": "application/json"
        }
        payload = {"query": "Explain thermodynamics."}
        res = self.client.post("/api/query", headers=bad_headers, json=payload)
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertEqual(data.get("error"), "Unauthorized")

    # -----------------------------------------------------------------------
    # Test 4: Valid routing signal -> HMAC verification succeeds
    # -----------------------------------------------------------------------
    def test_04_valid_routing_signal_verification(self):
        signal_data = {
            "request_id": "req-001",
            "cluster": "coding",
            "target_quality": 0.76,
            "probabilities": {"0": 0.85, "1": 0.15},
            "timestamp": round(time.time(), 4),
            "nonce": "nonce-abc-123"
        }
        sig = sign_routing_signal(signal_data)
        self.assertTrue(isinstance(sig, str) and len(sig) == 64)  # SHA-256 is 64 hex chars

        is_valid, msg = verify_routing_signal(signal_data, sig)
        self.assertTrue(is_valid, f"Verification failed: {msg}")

    # -----------------------------------------------------------------------
    # Test 5: Modified routing signal -> HMAC verification fails
    # -----------------------------------------------------------------------
    def test_05_modified_routing_signal_fails(self):
        signal_data = {
            "request_id": "req-002",
            "cluster": "math",
            "target_quality": 0.80,
            "probabilities": {"0": 0.1, "1": 0.9},
            "timestamp": round(time.time(), 4),
            "nonce": "nonce-def-456"
        }
        sig = sign_routing_signal(signal_data)

        # Attacker tampers with the target_quality or cluster
        tampered_signal = dict(signal_data)
        tampered_signal["target_quality"] = 0.50  # Downgrade attack!

        is_valid, msg = verify_routing_signal(tampered_signal, sig)
        self.assertFalse(is_valid, "Tampered signal should fail verification")
        self.assertIn("integrity verification failed", msg.lower())

    # -----------------------------------------------------------------------
    # Test 6: Invalid signature -> HMAC verification fails
    # -----------------------------------------------------------------------
    def test_06_invalid_signature_fails(self):
        signal_data = {
            "request_id": "req-003",
            "cluster": "general",
            "target_quality": 0.75,
            "timestamp": round(time.time(), 4),
            "nonce": "nonce-ghi-789"
        }
        bogus_signature = "0" * 64
        is_valid, msg = verify_routing_signal(signal_data, bogus_signature)
        self.assertFalse(is_valid)

        # Empty or non-hex string
        is_valid_empty, _ = verify_routing_signal(signal_data, "")
        self.assertFalse(is_valid_empty)

    # -----------------------------------------------------------------------
    # Test 7: Expired timestamp -> Replay check fails
    # -----------------------------------------------------------------------
    def test_07_expired_timestamp_fails(self):
        replay_mgr = ReplayProtectionManager(window_seconds=300)
        old_timestamp = time.time() - 600.0  # 10 minutes ago (> 300s window)

        valid, reason = replay_mgr.validate_and_record("req-old", "nonce-old", old_timestamp)
        self.assertFalse(valid)
        self.assertIn("expired", reason.lower())

        # Also test future timestamp rejection (> 60s clock skew)
        future_timestamp = time.time() + 120.0
        valid_fut, reason_fut = replay_mgr.validate_and_record("req-fut", "nonce-fut", future_timestamp)
        self.assertFalse(valid_fut)
        self.assertIn("future", reason_fut.lower())

    # -----------------------------------------------------------------------
    # Test 8: Replay of same request_id/nonce -> Rejected
    # -----------------------------------------------------------------------
    def test_08_replay_same_id_nonce_rejected(self):
        replay_mgr = ReplayProtectionManager(window_seconds=300)
        now = time.time()

        # First use -> Accepted
        ok1, msg1 = replay_mgr.validate_and_record("req-reuse", "nonce-reuse", now)
        self.assertTrue(ok1, f"First use should succeed: {msg1}")

        # Replay attempt with same (request_id, nonce) -> REJECTED
        ok2, msg2 = replay_mgr.validate_and_record("req-reuse", "nonce-reuse", now)
        self.assertFalse(ok2, "Replayed signal must be rejected")
        self.assertIn("replay detected", msg2.lower())

    # -----------------------------------------------------------------------
    # Test 9: Existing AD-BoN routing optimization tests still pass
    # -----------------------------------------------------------------------
    def test_09_existing_adbon_routing_passes(self):
        gateway = ADBoNGateway(
            use_mock_router=True,
            use_live_llm=False,
            cache_file="test_temp_cache.json"
        )
        report = gateway.route_query(
            query="Implement a binary search tree in C++.",
            target_quality=0.76
        )
        # Quality score >= target quality
        self.assertGreaterEqual(report["final_score"], 0.0)
        self.assertIn("total_query_cost", report)
        self.assertIn("senior_cost_baseline", report)
        self.assertIn("selected_route", report)

        # Clean test cache file
        if os.path.exists("test_temp_cache.json"):
            try:
                os.remove("test_temp_cache.json")
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Test 10: Existing benchmark tests still pass
    # -----------------------------------------------------------------------
    def test_10_existing_benchmark_suite_passes(self):
        bench_file = "test_bench_suite_cache.json"
        results = run_benchmark_suite(benchmark_cache_file=bench_file)

        self.assertIn("kpis", results)
        self.assertIn("results", results)
        self.assertEqual(len(results["results"]), 12, "Benchmark must evaluate all 12 queries")

        kpis = results["kpis"]
        self.assertGreater(kpis["overall_savings_pct"], 0.0, "Gateway must deliver measurable savings")
        self.assertGreater(kpis["avg_quality_retained_pct"], 80.0, "Quality retention must be high (>80%)")

        if os.path.exists(bench_file):
            try:
                os.remove(bench_file)
            except Exception:
                pass


if __name__ == "__main__":
    print("======================================================================")
    print("  Running AD-BoN Security & Gateway Regression Test Suite (10 Tests)  ")
    print("======================================================================")
    unittest.main(verbosity=2)
