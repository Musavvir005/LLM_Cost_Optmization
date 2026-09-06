"""
AD-BoN Gateway Engine Module
"""
import os
import sys

# Ensure root directory is on sys.path so modules can be imported smoothly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phase4_runtime_gateway_v4 import (
    ADBoNGateway,
    CONFIDENCE_THRESHOLD,
    CACHE_THRESHOLD,
    extract_file_metadata,
    create_sample_assets,
    MAX_FILE_SIZE_MB,
    MAX_FILE_SIZE_BYTES,
)

__all__ = [
    "ADBoNGateway",
    "CONFIDENCE_THRESHOLD",
    "CACHE_THRESHOLD",
    "extract_file_metadata",
    "create_sample_assets",
    "MAX_FILE_SIZE_MB",
    "MAX_FILE_SIZE_BYTES",
]
