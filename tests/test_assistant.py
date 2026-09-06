"""
Unit & Integration Test Suite for AD-BoN AI Assistant
=====================================================
Tests all 12 requirements specified for the AI Assistant integration:
  1. Assistant accepts a valid message.
  2. Empty message is rejected (HTTP 400).
  3. Missing authentication returns HTTP 401.
  4. Invalid authentication returns HTTP 401.
  5. Missing GROQ_API_KEY produces a clean configuration error.
  6. AD-BoN routing is actually called.
  7. Selected tier determines the Groq model.
  8. N=1 generates one candidate.
  9. N>1 generates multiple candidates.
 10. Response contains complete routing, cost, cache, and security metadata.
 11. API key never appears in the response.
 12. Existing AD-BoN regression tests still pass (verified in test_security.py).
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root and legacy gateway are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
LEGACY_DIR = os.path.join(PROJECT_ROOT, "backend", "gateway", "legacy")
for p in [PROJECT_ROOT, LEGACY_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["ADBON_API_KEY"] = "test-secret-api-key-xyz"
os.environ["ADBON_SIGNAL_SECRET"] = "test-signal-secret-987"
os.environ["GROQ_API_KEY"] = "mock_test_key_for_unit_tests"

from app import app as flask_app
from ai_assistant import AIAssistant, generate_with_groq, get_groq_model_mapping


class MockCompletionChoice:
    def __init__(self, content):
        self.message = MagicMock(content=content)


class MockCompletionUsage:
    def __init__(self, prompt_tokens=25, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockCompletionResponse:
    def __init__(self, content="Mock Groq answer"):
        self.choices = [MockCompletionChoice(content)]
        self.usage = MockCompletionUsage()


class TestAIAssistant(unittest.TestCase):

    def setUp(self):
        self.client = flask_app.test_client()
        self.api_key = "test-secret-api-key-xyz"
        self.auth_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # 1. Assistant accepts a valid message
    @patch("ai_assistant.generate_with_groq")
    def test_01_accepts_valid_message(self, mock_generate):
        mock_generate.return_value = (
            "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
            {"prompt_tokens": 30, "completion_tokens": 45, "total_tokens": 75}
        )
        resp = self.client.post(
            "/api/assistant/chat",
            headers=self.auth_headers,
            json={"message": "Write a Python function to check whether a number is prime."}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("prime number", data["answer"])
        self.assertIn("routing", data)
        self.assertIn("cost", data)

    # 2. Empty message is rejected
    def test_02_empty_message_rejected(self):
        for empty_val in ["", "   ", None]:
            resp = self.client.post(
                "/api/assistant/chat",
                headers=self.auth_headers,
                json={"message": empty_val}
            )
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertEqual(data["error"], "Bad Request")

    # 3. Missing authentication returns 401
    def test_03_missing_auth_returns_401(self):
        resp = self.client.post(
            "/api/assistant/chat",
            headers={"Content-Type": "application/json"},
            json={"message": "Explain recursion."}
        )
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertEqual(data["error"], "Unauthorized")

    # 4. Invalid authentication returns 401
    def test_04_invalid_auth_returns_401(self):
        resp = self.client.post(
            "/api/assistant/chat",
            headers={"Authorization": "Bearer invalid-wrong-token-abc", "Content-Type": "application/json"},
            json={"message": "Explain recursion."}
        )
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertEqual(data["error"], "Unauthorized")

    # 5. Missing GROQ_API_KEY produces a clean configuration error
    @patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False)
    def test_05_missing_groq_api_key_clean_error(self):
        # Create assistant with mock gateway
        mock_gw = MagicMock()
        mock_gw.route_query.return_value = {
            "selected_route": {"tier": "junior", "budget": 1, "expected_quality": 0.8},
            "primary_cluster_name": "general_knowledge",
            "is_prompt_cache_hit": False,
        }
        assistant = AIAssistant(gateway=mock_gw)
        with self.assertRaises(RuntimeError) as ctx:
            assistant.chat("Explain recursion.")
        self.assertIn("GROQ_API_KEY", str(ctx.exception))

    # 6. AD-BoN routing is actually called
    @patch("ai_assistant.generate_with_groq")
    def test_06_adbon_routing_is_called(self, mock_generate):
        mock_generate.return_value = ("Recursion is...", {"prompt_tokens": 10, "completion_tokens": 20})
        mock_gw = MagicMock()
        mock_gw.route_query.return_value = {
            "selected_route": {"tier": "junior", "budget": 1, "expected_quality": 0.82},
            "primary_cluster_name": "coding",
            "is_prompt_cache_hit": True,
        }
        assistant = AIAssistant(gateway=mock_gw)
        result = assistant.chat("Explain recursion in Python.")
        mock_gw.route_query.assert_called_once()
        self.assertEqual(result["routing"]["cluster"], "coding")
        self.assertEqual(result["routing"]["selected_tier"], "junior")

    # 7. Selected tier determines the Groq model
    @patch("ai_assistant.generate_with_groq")
    def test_07_tier_determines_groq_model(self, mock_generate):
        mock_generate.return_value = ("Test answer", {"prompt_tokens": 10, "completion_tokens": 10})
        mapping = get_groq_model_mapping()

        for tier in ["junior", "medium", "senior"]:
            mock_gw = MagicMock()
            mock_gw.route_query.return_value = {
                "selected_route": {"tier": tier, "budget": 1, "expected_quality": 0.9},
                "primary_cluster_name": "test",
                "is_prompt_cache_hit": False,
            }
            assistant = AIAssistant(gateway=mock_gw)
            res = assistant.chat("Test query")
            expected_model = mapping[tier]
            self.assertEqual(res["routing"]["groq_model"], expected_model)
            # Verify the call to generate_with_groq used this model
            args, kwargs = mock_generate.call_args
            self.assertEqual(kwargs.get("model"), expected_model)

    # 8. N=1 generates one candidate
    @patch("ai_assistant.generate_with_groq")
    def test_08_n_equals_1_generates_one_candidate(self, mock_generate):
        mock_generate.return_value = ("Answer N=1", {"prompt_tokens": 15, "completion_tokens": 25})
        mock_gw = MagicMock()
        mock_gw.route_query.return_value = {
            "selected_route": {"tier": "junior", "budget": 1, "expected_quality": 0.78},
            "primary_cluster_name": "factual",
            "is_prompt_cache_hit": False,
        }
        assistant = AIAssistant(gateway=mock_gw)
        res = assistant.chat("What is the capital of France?")
        self.assertEqual(mock_generate.call_count, 1)
        self.assertEqual(res["routing"]["budget"], 1)

    # 9. N>1 generates multiple candidates
    @patch("ai_assistant.generate_with_groq")
    def test_09_n_greater_than_1_generates_multiple_candidates(self, mock_generate):
        mock_generate.side_effect = [
            ("Short candidate 1", {"prompt_tokens": 20, "completion_tokens": 30}),
            ("Longer and much more comprehensive candidate 2 with code example", {"prompt_tokens": 20, "completion_tokens": 60}),
            ("Candidate 3", {"prompt_tokens": 20, "completion_tokens": 25}),
        ]
        mock_gw = MagicMock()
        mock_gw.route_query.return_value = {
            "selected_route": {"tier": "medium", "budget": 3, "expected_quality": 0.92},
            "primary_cluster_name": "coding",
            "is_prompt_cache_hit": False,
        }
        assistant = AIAssistant(gateway=mock_gw)
        res = assistant.chat("Implement a red-black tree.")
        self.assertEqual(mock_generate.call_count, 3)
        self.assertEqual(res["routing"]["budget"], 3)
        # Verify it selected the best/most complete candidate
        self.assertIn("Longer and much more comprehensive", res["answer"])

    # 10. Response contains routing metadata
    @patch("ai_assistant.generate_with_groq")
    def test_10_response_contains_full_metadata(self, mock_generate):
        mock_generate.return_value = ("Detailed response", {"prompt_tokens": 50, "completion_tokens": 100})
        mock_gw = MagicMock()
        mock_gw.route_query.return_value = {
            "selected_route": {"tier": "medium", "budget": 1, "expected_quality": 0.85},
            "primary_cluster_name": "math",
            "is_prompt_cache_hit": True,
        }
        assistant = AIAssistant(gateway=mock_gw)
        res = assistant.chat("Solve the integral.")

        # Check all required top-level keys
        self.assertIn("answer", res)
        self.assertIn("routing", res)
        self.assertIn("cost", res)
        self.assertIn("cache", res)
        self.assertIn("performance", res)
        self.assertIn("security", res)

        # Check nested metadata
        self.assertEqual(res["routing"]["cluster"], "math")
        self.assertEqual(res["routing"]["selected_tier"], "medium")
        self.assertIn("groq_model", res["routing"])
        self.assertIn("budget", res["routing"])
        self.assertIn("expected_quality", res["routing"])
        self.assertIn("target_quality", res["routing"])

        self.assertIn("estimated_cost", res["cost"])
        self.assertIn("baseline_cost", res["cost"])
        self.assertIn("savings_percent", res["cost"])

        self.assertEqual(res["cache"]["hit"], True)
        self.assertGreater(res["cache"]["tokens_saved"], 0)

        self.assertIn("latency_ms", res["performance"])
        self.assertTrue(res["security"]["authenticated"])

    # 11. API key never appears in the response
    @patch("ai_assistant.generate_with_groq")
    def test_11_api_key_never_appears_in_response(self, mock_generate):
        mock_generate.return_value = ("Safe response.", {"prompt_tokens": 10, "completion_tokens": 10})
        secret_groq = os.environ.get("GROQ_API_KEY", "")
        secret_adbon = os.environ.get("ADBON_API_KEY", "")

        resp = self.client.post(
            "/api/assistant/chat",
            headers=self.auth_headers,
            json={"message": "Tell me a joke."}
        )
        self.assertEqual(resp.status_code, 200)
        raw_text = resp.get_data(as_text=True)

        self.assertNotIn(secret_groq, raw_text)
        self.assertNotIn(secret_adbon, raw_text)
        self.assertNotIn("Bearer", raw_text)


if __name__ == "__main__":
    print("=" * 70)
    print("  Running AD-BoN AI Assistant Test Suite (11 Core Unit Tests)  ")
    print("=" * 70)
    unittest.main()
