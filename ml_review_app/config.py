"""Runtime configuration for the Flask ML review app."""

from __future__ import annotations

import os
from pathlib import Path


class Config:
    """Default local configuration."""

    SECRET_KEY = os.environ.get("ML_REVIEW_SECRET_KEY", "dev-secret-change-me")
    RUNTIME_DIR = Path(os.environ.get("ML_REVIEW_RUNTIME_DIR", "runtime")).resolve()
    MAX_CONTENT_LENGTH = int(os.environ.get("ML_REVIEW_MAX_UPLOAD_MB", "100")) * 1024 * 1024
    MAX_PROJECT_NAME_LENGTH = 200
    MAX_SEARCH_STRATEGY_LENGTH = 10_000
    MAX_INCLUSION_CRITERIA_LENGTH = 50_000
    MAX_API_KEY_LENGTH = 512
    MAX_CLUSTER_COUNT = 20
    MAX_PUBMED_RECORDS = 10_000
    DEFAULT_EMBEDDING_MODEL = os.environ.get("ML_REVIEW_EMBEDDING_MODEL", "text-embedding-3-small")
    DEFAULT_SCREENING_MODEL = os.environ.get("ML_REVIEW_SCREENING_MODEL", "gpt-5.6-luna")
    DEFAULT_EXTRACTION_MODEL = os.environ.get("ML_REVIEW_EXTRACTION_MODEL", "gpt-5.6-luna")
    DEFAULT_EMBEDDING_BATCH_SIZE = 100
    DEFAULT_TSNE_RANDOM_STATE = 42
    PUBLIC_BASE_URL = os.environ.get("ML_REVIEW_PUBLIC_BASE_URL", "").strip()
    BACKGROUND_TASKS_EAGER = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    TESTING = False


class TestConfig(Config):
    """Configuration used by the automated test suite."""

    TESTING = True
    WTF_CSRF_ENABLED = False
    BACKGROUND_TASKS_EAGER = True
