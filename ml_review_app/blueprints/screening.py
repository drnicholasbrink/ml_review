"""AI screening routes using local deterministic screening by default."""

from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, url_for

from ..services.project_service import load_manifest, project_dir, save_manifest
from ..services.screening_service import screen_csv_offline

bp = Blueprint("screening", __name__, url_prefix="/projects/<project_id>/screening")


@bp.route("", methods=["GET", "POST"])
def screening(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    criteria_path = path / "inclusion_criteria.txt"
    criteria = criteria_path.read_text() if criteria_path.exists() else ""
    rows = None
    if request.method == "POST":
        source = manifest.get("files", {}).get("selected_records") or manifest.get("files", {}).get("labeled_clusters") or manifest.get("files", {}).get("deduplicated_records")
        df = screen_csv_offline(path / source, criteria, path / "ai_screening_full_results.csv")
        manifest.setdefault("files", {})["ai_screening_full_results"] = "ai_screening_full_results.csv"
        manifest["screening_rows"] = len(df)
        manifest["screening_decision_counts"] = df["ai_decision"].value_counts().to_dict()
        save_manifest(path, manifest)
        rows = df.head(100).to_dict(orient="records")
    return render_template("screening.html", manifest=manifest, criteria=criteria, rows=rows)
