"""
AD-BoN Security Layer
=====================
Lightweight API authentication, HMAC-SHA256 routing-signal integrity,
and replay protection for the AD-BoN Runtime Gating Gateway.

Security Principles:
  - Standard Bearer token authentication with constant-time verification.
  - HMAC-SHA256 routing signal signing over canonicalized JSON representations.
  - Replay protection with timestamp TTL windows and request_id/nonce deduplication.
  - Zero sensitive credentials printed in logs or returned in API responses.
  - Clean modular helpers easily adaptable to Flask, FastAPI, or future OAuth/JWT.
"""

import os
import json
import time
import hmac
import hashlib
import secrets
from typing import Dict, Any, Tuple, Optional
from functools import wraps

# ---------------------------------------------------------------------------
# Configuration (Environment-backed with safe development defaults)
# ---------------------------------------------------------------------------
DEFAULT_DEV_API_KEY = "adbon-sec-key-2026-demo"
DEFAULT_DEV_SIGNAL_SECRET = "adbon-signal-hmac-secret-2026"
DEFAULT_REPLAY_WINDOW = 300  # 5 minutes in seconds

def get_expected_api_key() -> str:
    return os.getenv("ADBON_API_KEY", DEFAULT_DEV_API_KEY)

def get_signal_secret() -> str:
    return os.getenv("ADBON_SIGNAL_SECRET", DEFAULT_DEV_SIGNAL_SECRET)

def get_replay_window_seconds() -> int:
    try:
        return int(os.getenv("ADBON_REPLAY_WINDOW", str(DEFAULT_REPLAY_WINDOW)))
    except ValueError:
        return DEFAULT_REPLAY_WINDOW


# ---------------------------------------------------------------------------
# 1. API Authentication
# ---------------------------------------------------------------------------
def verify_api_key(auth_header: Optional[str], expected_key: Optional[str] = None) -> bool:
    """
    Validates standard 'Authorization: Bearer <API_KEY>' header in constant time.
    Returns True if valid, False otherwise.
    """
    if not auth_header:
        return False

    parts = auth_header.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    provided_token = parts[1]
    target_key = expected_key if expected_key is not None else get_expected_api_key()

    # Use constant-time comparison to protect against timing side-channel attacks
    return hmac.compare_digest(provided_token.encode("utf-8"), target_key.encode("utf-8"))


# ---------------------------------------------------------------------------
# 2. Canonicalization & HMAC-SHA256 Routing Signal Protection
# ---------------------------------------------------------------------------
def canonicalize_signal(signal: Dict[str, Any]) -> bytes:
    """
    Produces deterministic canonical JSON bytes (sorted keys, compact separators).
    """
    return json.dumps(signal, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_routing_signal(signal: Dict[str, Any], secret: Optional[str] = None) -> str:
    """
    Calculates an HMAC-SHA256 signature over the canonical JSON representation
    of the routing signal.
    """
    key = (secret if secret is not None else get_signal_secret()).encode("utf-8")
    message = canonicalize_signal(signal)
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_routing_signal(
    signal: Dict[str, Any],
    signature: str,
    secret: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Verifies the HMAC-SHA256 signature of a routing signal in constant time.
    Returns (is_valid, reason).
    """
    if not signature or not isinstance(signature, str):
        return False, "Missing or invalid signature format"

    expected_sig = sign_routing_signal(signal, secret=secret)
    if not hmac.compare_digest(expected_sig.encode("utf-8"), signature.encode("utf-8")):
        return False, "Routing signal integrity verification failed"

    return True, "Integrity verified"


# ---------------------------------------------------------------------------
# 3. Replay Protection Manager
# ---------------------------------------------------------------------------
class ReplayProtectionManager:
    """
    In-memory replay protection cache.
    Tracks seen (request_id, nonce) combinations and validates signal timestamps.
    Automatically purges entries older than the replay window.
    """

    def __init__(self, window_seconds: Optional[int] = None):
        self.window_seconds = window_seconds or get_replay_window_seconds()
        # Map (request_id, nonce) -> timestamp
        self._seen_signals: Dict[Tuple[str, str], float] = {}

    def _cleanup_expired(self, current_time: float) -> None:
        cutoff = current_time - self.window_seconds
        expired_keys = [k for k, ts in self._seen_signals.items() if ts < cutoff]
        for k in expired_keys:
            del self._seen_signals[k]

    def validate_and_record(
        self,
        request_id: str,
        nonce: str,
        timestamp: float,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validates timestamp window and nonce uniqueness.
        Returns (is_valid, reason).
        """
        now = current_time if current_time is not None else time.time()
        self._cleanup_expired(now)

        # 1. Timestamp validity check
        if timestamp > now + 60.0:  # Allow 60s max future clock skew
            return False, "Routing signal timestamp is set in the future"

        if now - timestamp > self.window_seconds:
            return False, f"Routing signal expired (outside allowed {self.window_seconds}s window)"

        # 2. Replay check (request_id + nonce)
        key = (str(request_id), str(nonce))
        if key in self._seen_signals:
            return False, f"Replay detected: request_id/nonce combination already used"

        # Record valid signal
        self._seen_signals[key] = timestamp
        return True, "Replay check passed"

    def clear(self) -> None:
        """Reset internal cache (useful for test isolation)."""
        self._seen_signals.clear()


# Shared singleton instance
_replay_manager = ReplayProtectionManager()

def get_replay_manager() -> ReplayProtectionManager:
    return _replay_manager


# ---------------------------------------------------------------------------
# 4. Helper: Create Secure Routing Signal Envelope
# ---------------------------------------------------------------------------
def create_secure_routing_signal(routing_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Enriches routing information with request_id, timestamp, and cryptographic nonce,
    then returns (signal, signature).
    """
    signal = dict(routing_payload)
    signal["request_id"] = signal.get("request_id", secrets.token_hex(12))
    signal["timestamp"] = signal.get("timestamp", round(time.time(), 4))
    signal["nonce"] = signal.get("nonce", secrets.token_hex(8))

    signature = sign_routing_signal(signal)
    return signal, signature


# ---------------------------------------------------------------------------
# 5. Flask Authentication Decorator
# ---------------------------------------------------------------------------
def require_flask_auth(f):
    """
    Flask route decorator that enforces Bearer token authentication.
    Returns clean 401 JSON error on failure without leaking credentials.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            from flask import request, jsonify
        except ImportError:
            raise RuntimeError("Flask is required to use @require_flask_auth decorator.")

        auth_header = request.headers.get("Authorization")
        if not verify_api_key(auth_header):
            return jsonify({
                "error": "Unauthorized",
                "message": "Valid API credentials are required."
            }), 401

        return f(*args, **kwargs)
    return decorated_function
