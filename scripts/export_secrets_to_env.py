"""Export ignored Python secrets to a gitignored Compose .env file."""

from __future__ import annotations

import importlib.util
import os
import secrets
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = Path(__file__).with_name("secret_keys.py")
OUTPUT_PATH = ROOT / ".env"
KEY_NAMES = ("OPENAI_API_KEY", "PUBMED_API_KEY")


def _existing_secret_key() -> str:
    if not OUTPUT_PATH.exists():
        return ""
    for line in OUTPUT_PATH.read_text().splitlines():
        if line.startswith("ML_REVIEW_SECRET_KEY="):
            return line.partition("=")[2].strip()
    return ""


def main() -> None:
    if not SECRETS_PATH.is_file():
        raise SystemExit(f"Secrets file not found: {SECRETS_PATH}")
    spec = importlib.util.spec_from_file_location("ml_review_local_secrets", SECRETS_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load secret_keys.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in KEY_NAMES if not str(getattr(module, name, "")).strip()]
    if missing:
        raise SystemExit(f"Missing values in secret_keys.py: {', '.join(missing)}")
    flask_secret = _existing_secret_key() or secrets.token_urlsafe(48)
    lines = [f"ML_REVIEW_SECRET_KEY={flask_secret}"]
    lines.extend(f"{name}={str(getattr(module, name)).strip()}" for name in KEY_NAMES)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    os.chmod(OUTPUT_PATH, 0o600)
    print(f"Wrote {OUTPUT_PATH.name} with permissions 0600")


if __name__ == "__main__":
    main()
