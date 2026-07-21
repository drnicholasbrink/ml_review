"""Home and health routes."""

from __future__ import annotations

import os
from pathlib import Path

import plotly
from flask import Blueprint, current_app, render_template, send_file

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/health")
def health():
    return {"status": "ok"}


@bp.get("/ready")
def ready():
    runtime_dir = current_app.config["RUNTIME_DIR"]
    ready_state = runtime_dir.is_dir() and os.access(runtime_dir, os.R_OK | os.W_OK)
    return {"status": "ready" if ready_state else "not_ready", "runtime_writable": ready_state}, 200 if ready_state else 503


@bp.get("/vendor/plotly.min.js")
def plotly_javascript():
    """Serve the installed Plotly bundle without an external runtime dependency."""

    bundle = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return send_file(bundle, mimetype="text/javascript", max_age=3_600)
