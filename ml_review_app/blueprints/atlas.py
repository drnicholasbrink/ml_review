"""Evidence Atlas page, build, and artifact routes."""

from __future__ import annotations

import base64
import json
import re
import zlib
from pathlib import Path
from urllib.parse import urlencode, urlparse

from flask import Blueprint, Response, current_app, redirect, render_template, request, send_file, url_for

from ..services.atlas_service import ATLAS_DIRECTORY, ATLAS_VERSION, atlas_status, build_atlas
from ..services.project_service import load_manifest, project_dir

bp = Blueprint("atlas", __name__, url_prefix="/projects/<project_id>/atlas")
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
APPLE_ATLAS_APP_URL = "https://apple.github.io/embedding-atlas/app/"
APPLE_ATLAS_ORIGIN = "https://apple.github.io"


def _render(project_id: str, *, error: str | None = None, status_code: int = 200):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    status = atlas_status(path, manifest)
    preload_available = False
    if status.get("state") == "ready":
        preload_available = _preload_available()
    return (
        render_template(
            "atlas.html",
            manifest=manifest,
            atlas_status=status,
            atlas_preload_available=preload_available,
            apple_atlas_app_url=APPLE_ATLAS_APP_URL,
            error=error,
        ),
        status_code,
    )


@bp.get("")
def atlas(project_id: str):
    return _render(project_id)


@bp.post("/build")
def build(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        build_atlas(path, manifest)
    except ValueError as exc:
        return _render(project_id, error=str(exc), status_code=400)
    return redirect(url_for("atlas.atlas", project_id=project_id))


def _current_artifact(project_id: str, fingerprint: str) -> Path | None:
    if not FINGERPRINT_PATTERN.fullmatch(fingerprint):
        return None
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    status = atlas_status(path, manifest)
    record = status.get("record") or {}
    artifact = path / ATLAS_DIRECTORY / f"{fingerprint}.parquet"
    if status.get("state") != "ready" or record.get("fingerprint") != fingerprint or not artifact.is_file():
        return None
    return artifact


@bp.get("/data/<fingerprint>.parquet")
def data(project_id: str, fingerprint: str):
    artifact = _current_artifact(project_id, fingerprint)
    if artifact is None:
        return Response(status=404)
    response = send_file(artifact, mimetype="application/vnd.apache.parquet", conditional=True)
    response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
    return response


def _settings_payload() -> str:
    settings = {
        "version": ATLAS_VERSION,
        "text": "search_text",
        "embedding": {
            "precomputed": {"x": "atlas_x", "y": "atlas_y", "neighbors": "neighbors"},
        },
    }
    compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(json.dumps(settings, separators=(",", ":")).encode("utf-8"))
    compressed += compressor.flush()
    return base64.urlsafe_b64encode(compressed).rstrip(b"=").decode("ascii")


def _public_artifact_url(project_id: str, fingerprint: str) -> str:
    path = url_for("atlas.data", project_id=project_id, fingerprint=fingerprint)
    public_base_url = str(current_app.config.get("PUBLIC_BASE_URL") or "").strip()
    return f"{public_base_url.rstrip('/')}{path}" if public_base_url else url_for(
        "atlas.data", project_id=project_id, fingerprint=fingerprint, _external=True
    )


def _preload_available() -> bool:
    public_base_url = str(current_app.config.get("PUBLIC_BASE_URL") or "").strip()
    parsed = urlparse(public_base_url)
    return parsed.scheme == "https" and bool(parsed.netloc)


@bp.get("/open/<fingerprint>")
def open_in_apple(project_id: str, fingerprint: str):
    """Open Apple's official file viewer with this Atlas artifact and saved column settings."""

    if _current_artifact(project_id, fingerprint) is None:
        return Response(status=404)
    artifact_url = _public_artifact_url(project_id, fingerprint)
    if not _preload_available():
        return _render(
            project_id,
            error=(
                "One-click preload requires a publicly reachable HTTPS address. "
                "Download the Atlas Parquet file and open it in the official viewer instead."
            ),
            status_code=409,
        )
    fragment = urlencode(
        {"data": artifact_url, "settings": _settings_payload()}
    )
    return redirect(f"{APPLE_ATLAS_APP_URL}#?{fragment}")


@bp.after_request
def atlas_headers(response: Response) -> Response:
    if request.endpoint == "atlas.data":
        response.headers["Access-Control-Allow-Origin"] = APPLE_ATLAS_ORIGIN
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        response.headers.add("Vary", "Origin")
    else:
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self' data:; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    return response
