"""Interactive screening evaluation and human-reference comparison routes."""

from __future__ import annotations

import json

import pandas as pd
from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.evaluation_service import (
    build_screening_evaluation,
    compare_with_human,
    workflow_human_reference,
)
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
    workflow_reference = None
    files = manifest.get("files", {})
    workflow_name = files.get("full_text_screening_results") or files.get("human_screening_reviewed_results")
    if workflow_name and (path / workflow_name).is_file():
        try:
            workflow_reference = workflow_human_reference(pd.read_csv(path / workflow_name))
        except ValueError:
            workflow_reference = None

    if request.method == "POST":
        try:
            if screening_df is None:
                raise ValueError("Run screening before comparing against human decisions")
            reference_source = request.form.get("reference_source", "upload")
            if reference_source not in {"workflow", "upload"}:
                raise ValueError("Choose workflow decisions or an uploaded CSV as the human reference")
            threshold = parse_bounded_int(
                request.form.get("threshold"),
                "Title match threshold",
                minimum=0,
                maximum=100,
                default=85,
            )
            uncertain_is_positive = request.form.get("uncertain_is_positive") == "on"
            if reference_source == "workflow":
                if workflow_reference is None:
                    raise ValueError("Record at least one human screening decision before using workflow decisions")
                human_df = workflow_reference
                count_unmatched_ai = False
                reference_label = "Human decisions recorded in the review workflow"
            else:
                upload = request.files.get("human_screening_csv")
                if not upload or not upload.filename:
                    raise ValueError("Choose an included-records CSV to evaluate")
                if not upload.filename.lower().endswith(".csv"):
                    raise ValueError("Human screening reference must be a .csv file")
                human_path = path / "human_screening_reference.csv"
                human_df = pd.read_csv(upload.stream)
                if human_df.empty:
                    raise ValueError("The human-screening CSV is empty")
                upload.stream.seek(0)
                upload.save(human_path)
                count_unmatched_ai = True
                reference_label = "Uploaded included-records CSV"
            metrics, comparison = compare_with_human(
                screening_df,
                human_df,
                threshold=threshold,
                uncertain_is_positive=uncertain_is_positive,
                count_unmatched_ai=count_unmatched_ai,
            )
            metrics["reference_source"] = reference_source
            metrics["reference_label"] = reference_label
            metrics["workflow_reviewed_records"] = len(human_df) if reference_source == "workflow" else None
            metrics_path = path / "human_evaluation_metrics.json"
            comparison_path = path / "human_evaluation_comparison.csv"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            comparison.to_csv(comparison_path, index=False)
            manifest_files = manifest.setdefault("files", {})
            if reference_source == "upload":
                manifest_files["human_screening_reference"] = human_path.name
            else:
                manifest_files.pop("human_screening_reference", None)
            manifest_files["human_evaluation_metrics"] = metrics_path.name
            manifest_files["human_evaluation_comparison"] = comparison_path.name
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
        workflow_reference_count=0 if workflow_reference is None else len(workflow_reference),
        error=error,
    ), 400 if error else 200
