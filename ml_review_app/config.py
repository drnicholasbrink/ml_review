"""Runtime configuration for the Flask ML review app."""

from __future__ import annotations

import os
from pathlib import Path


class Config:
    """Default local configuration."""

    SECRET_KEY = os.environ.get("ML_REVIEW_SECRET_KEY", "dev-secret-change-me")
    RUNTIME_DIR = Path(os.environ.get("ML_REVIEW_RUNTIME_DIR", "runtime")).resolve()
    MAX_CONTENT_LENGTH = int(os.environ.get("ML_REVIEW_MAX_UPLOAD_MB", "100")) * 1024 * 1024
    TESTING = False


class TestConfig(Config):
    """Configuration used by the automated test suite."""

    TESTING = True
    WTF_CSRF_ENABLED = False
