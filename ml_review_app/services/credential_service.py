"""Resolve API credentials without persisting user-submitted values."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def _load_local_secrets() -> ModuleType | None:
    candidates = [
        Path.cwd() / "secret_keys.py",
        Path.cwd() / "scripts" / "secret_keys.py",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("ml_review_local_secrets", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def resolve_api_key(submitted: str | None, name: str, *, required: bool) -> tuple[str, str]:
    """Resolve a submitted, environment, or local Python API key in that order."""

    submitted_value = (submitted or "").strip()
    if submitted_value:
        return submitted_value, "entered for this request"
    environment_value = os.environ.get(name, "").strip()
    if environment_value:
        return environment_value, "configured environment"
    local_secrets = _load_local_secrets()
    local_value = str(getattr(local_secrets, name, "")).strip() if local_secrets else ""
    if local_value:
        return local_value, "local secret_keys.py"
    if required:
        raise ValueError(f"{name} is not configured. Enter a key or configure it in .env.")
    return "", "anonymous access"


def credential_available(name: str) -> bool:
    """Return whether a fallback credential exists without exposing it."""

    try:
        value, _source = resolve_api_key(None, name, required=False)
    except (ImportError, OSError, ValueError):
        return False
    return bool(value)
