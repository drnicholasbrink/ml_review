"""Structured extraction routes for included studies."""

from __future__ import annotations

import json

import pandas as pd
from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.credential_service import credential_available, resolve_api_key
from ..services.extraction_service import extract_csv, extraction_candidates, write_extraction_exports
from ..services.project_service import load_manifest, project_dir, save_manifest
from ..services.validation_service import parse_bounded_int, validate_text

bp = Blueprint("extraction", __name__, url_prefix="/projects/<project_id>/extraction")


def _preview_rows(frame: pd.DataFrame) -> list[dict]:
    columns = [
        "RecordID", "PMID", "Title", "country", "study_design", "sample_size", "extraction_confidence",
        "data_completeness", "key_finding_summary", "effect_estimates", "notes",
    ]
    preview = frame[[column for column in columns if column in frame.columns]].head(250).copy()
    if "effect_estimates" in preview:
        def list_length(value):
            try:
                parsed = json.loads(value or "[]")
                return len(parsed) if isinstance(parsed, list) else 0
            except (json.JSONDecodeError, TypeError):
                return 0
        preview["effect_estimate_count"] = preview["effect_estimates"].fillna("[]").map(list_length)
    return preview.fillna("").to_dict(orient="records")


@bp.route("", methods=["GET", "POST"])
def extraction(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    files = manifest.get("files", {})
    screening_name = files.get("human_screening_reviewed_results") or files.get("ai_screening_full_results")
    decision_source_label = "Human-reviewed final decisions" if files.get("human_screening_reviewed_results") else "AI screening decisions"
    screening_path = path / screening_name if screening_name else None
    source_ready = bool(screening_path and screening_path.is_file())
    criteria_path = path / "inclusion_criteria.txt"
    criteria = criteria_path.read_text() if criteria_path.is_file() else ""
    fallback_available = credential_available("OPENAI_API_KEY")
    error = None
    rows = None
    candidate_count = 0
    include_uncertain = bool(manifest.get("extraction_include_uncertain", False))

    if source_ready:
        try:
            candidate_count = len(extraction_candidates(pd.read_csv(screening_path), include_uncertain=include_uncertain))
        except ValueError:
            candidate_count = 0

    if request.method == "POST":
        try:
            if not source_ready:
                raise ValueError("Run screening before extracting study data")
            criteria = validate_text(
                criteria,
                "Inclusion/exclusion criteria",
                current_app.config["MAX_INCLUSION_CRITERIA_LENGTH"],
                required=True,
            )
            submitted_key = validate_text(
                request.form.get("openai_api_key"),
                "OpenAI API key",
                current_app.config["MAX_API_KEY_LENGTH"],
            )
            api_key, key_source = resolve_api_key(submitted_key, "OPENAI_API_KEY", required=True)
            model = validate_text(request.form.get("model"), "Extraction model", 100, required=True)
            allowed_models = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
            if model not in allowed_models:
                raise ValueError("Choose a supported OpenAI extraction model")
            scope = request.form.get("scope", "test")
            if scope not in {"test", "full"}:
                raise ValueError("Choose a test or full extraction run")
            test_limit = parse_bounded_int(
                request.form.get("test_limit"),
                "Test records",
                minimum=1,
                maximum=25,
                default=5,
            )
            include_uncertain = request.form.get("include_uncertain") == "on"
            output_path = path / "ai_extraction_full_results.csv"
            limit = test_limit if scope == "test" else None
            resume = request.form.get("resume") == "on"

            def run_extraction(progress):
                result, total_candidates = extract_csv(
                    screening_path,
                    criteria,
                    output_path,
                    api_key=api_key,
                    model=model,
                    include_uncertain=include_uncertain,
                    limit=limit,
                    resume=resume,
                    progress_callback=progress,
                )
                derived = write_extraction_exports(result, path)
                updated_manifest = load_manifest(path)
                updated_manifest.setdefault("files", {})["ai_extraction_full_results"] = output_path.name
                updated_manifest["files"].update(derived)
                updated_manifest["extraction_rows"] = len(result)
                updated_manifest["extraction_candidate_rows"] = total_candidates
                updated_manifest["extraction_confidence_counts"] = result["extraction_confidence"].value_counts().to_dict()
                updated_manifest["extraction_completeness_counts"] = result["data_completeness"].value_counts().to_dict()
                updated_manifest["extraction_model"] = model
                updated_manifest["extraction_include_uncertain"] = include_uncertain
                updated_manifest["extraction_last_scope"] = scope
                updated_manifest["openai_key_source"] = key_source
                save_manifest(path, updated_manifest)

            task = current_app.extensions["task_manager"].submit(
                path,
                kind="extraction",
                title="Run structured extraction",
                target=run_extraction,
                result_url=url_for("extraction.extraction", project_id=project_id),
                failure_message="OpenAI extraction failed. Check the key, account access, and article content, then resume the task.",
            )
            return redirect(url_for("extraction.extraction", project_id=project_id, task=task["task_id"]))
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)

    if rows is None:
        output_name = manifest.get("files", {}).get("ai_extraction_full_results")
        if output_name and (path / output_name).is_file():
            rows = _preview_rows(pd.read_csv(path / output_name))
    return render_template(
        "extraction.html",
        manifest=manifest,
        criteria=criteria,
        source_ready=source_ready,
        fallback_available=fallback_available,
        candidate_count=candidate_count,
        include_uncertain=include_uncertain,
        decision_source_label=decision_source_label,
        rows=rows,
        error=error,
    ), 400 if error else 200
