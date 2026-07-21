"""Interactive screening evaluation and human-reference comparison routes."""

from __future__ import annotations

import json

import pandas as pd
from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.evaluation_service import build_screening_evaluation, compare_with_human
from ..services.project_service import load_manifest, project_dir, save_manifest
from ..services.validation_service import parse_bounded_int

bp = Blueprint("evaluation", __name__, url_prefix="/projects/<project_id>/evaluation")


def _load_evaluation(path, manifest):
    screening_name = manifest.get("files", {}).get("ai_screening_full_results")
    if not screening_name or not (path / screening_name).exists():
        return None, None
    screening_df = pd.read_csv(path / screening_name)
    return screening_df, build_screening_evaluation(screening_df)


@bp.route("", methods=["GET", "POST"])
def evaluation(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    screening_df, evaluation_data = _load_evaluation(path, manifest)
    error = None

    if request.method == "POST":
        try:
            if screening_df is None:
                raise ValueError("Run screening before comparing against human decisions")
            upload = request.files.get("human_screening_csv")
            if not upload or not upload.filename:
                raise ValueError("Choose a human-screening CSV to evaluate")
            if not upload.filename.lower().endswith(".csv"):
                raise ValueError("Human screening reference must be a .csv file")
            threshold = parse_bounded_int(
                request.form.get("threshold"),
                "Title match threshold",
                minimum=0,
                maximum=100,
                default=85,
            )
            uncertain_is_positive = request.form.get("uncertain_is_positive") == "on"
            human_path = path / "human_screening_reference.csv"
            human_df = pd.read_csv(upload.stream)
            if human_df.empty:
                raise ValueError("The human-screening CSV is empty")
            upload.stream.seek(0)
            upload.save(human_path)
            metrics, comparison = compare_with_human(
                screening_df,
                human_df,
                threshold=threshold,
                uncertain_is_positive=uncertain_is_positive,
            )
            metrics_path = path / "human_evaluation_metrics.json"
            comparison_path = path / "human_evaluation_comparison.csv"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            comparison.to_csv(comparison_path, index=False)
            manifest.setdefault("files", {})["human_screening_reference"] = human_path.name
            manifest["files"]["human_evaluation_metrics"] = metrics_path.name
            manifest["files"]["human_evaluation_comparison"] = comparison_path.name
            manifest["human_evaluation"] = metrics
            save_manifest(path, manifest)
            return redirect(url_for("evaluation.evaluation", project_id=project_id))
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            error = str(exc)

    metrics = manifest.get("human_evaluation")
    comparison_rows = None
    comparison_name = manifest.get("files", {}).get("human_evaluation_comparison")
    if comparison_name and (path / comparison_name).exists():
        comparison_rows = pd.read_csv(path / comparison_name).head(250).fillna("").to_dict(orient="records")
    return render_template(
        "evaluation.html",
        manifest=manifest,
        evaluation=evaluation_data,
        metrics=metrics,
        comparison_rows=comparison_rows,
        error=error,
    ), 400 if error else 200
