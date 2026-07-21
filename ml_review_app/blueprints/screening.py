"""OpenAI screening and human adjudication routes."""

from __future__ import annotations

import math

import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..services.credential_service import credential_available, resolve_api_key
from ..services.project_service import invalidate_extraction, load_manifest, project_dir, save_manifest
from ..services.screening_service import apply_human_reviews, save_human_review, screen_csv
from ..services.validation_service import validate_text

bp = Blueprint("screening", __name__, url_prefix="/projects/<project_id>/screening")
PAGE_SIZE = 100


def _preview_rows(df: pd.DataFrame) -> list[dict]:
    columns = [
        "record_key", "RecordID", "PMID", "DOI", "Title", "Abstract", "ai_decision", "ai_confidence",
        "ai_exclusion_reason", "ai_population_match", "ai_exposure_match", "ai_outcome_match",
        "ai_study_design_appropriate", "ai_reasoning", "ai_input_truncated", "human_decision",
        "human_note", "human_reviewed_at", "final_decision", "final_decision_source",
        "requires_human_review",
    ]
    preview = df[[column for column in columns if column in df.columns]].copy()
    for column in (
        "ai_population_match", "ai_exposure_match", "ai_outcome_match",
        "ai_study_design_appropriate", "ai_input_truncated", "requires_human_review",
    ):
        if column in preview:
            preview[column] = preview[column].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return preview.fillna("").to_dict(orient="records")


def _load_reviewed_results(path, manifest: dict) -> pd.DataFrame | None:
    results_name = manifest.get("files", {}).get("ai_screening_full_results")
    if not results_name or not (path / results_name).exists():
        return None
    screening_df = pd.read_csv(path / results_name)
    reviews_name = manifest.get("files", {}).get("human_screening_decisions")
    reviews_df = pd.read_csv(path / reviews_name, dtype=str) if reviews_name and (path / reviews_name).exists() else None
    return apply_human_reviews(screening_df, reviews_df)


def _filtered_page(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    review_filter = request.args.get("review_filter", "all")
    if review_filter not in {"all", "needs_review", "reviewed"}:
        review_filter = "all"
    query = request.args.get("q", "").strip()[:300]
    filtered = df
    if review_filter == "needs_review":
        filtered = filtered.loc[filtered["requires_human_review"]]
    elif review_filter == "reviewed":
        filtered = filtered.loc[filtered["final_decision_source"] == "human"]
    if query:
        searchable = filtered.reindex(columns=["RecordID", "PMID", "Title", "ai_decision", "ai_exclusion_reason", "ai_reasoning"]).fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered.loc[searchable.str.contains(query, case=False, regex=False)]
    total = len(filtered)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    try:
        page = max(1, min(int(request.args.get("page", 1)), pages))
    except ValueError:
        page = 1
    start = (page - 1) * PAGE_SIZE
    return filtered.iloc[start:start + PAGE_SIZE], {
        "q": query, "review_filter": review_filter, "page": page, "pages": pages,
        "filtered_count": total, "start": start + 1 if total else 0, "end": min(start + PAGE_SIZE, total),
    }


@bp.route("", methods=["GET", "POST"])
def screening(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    criteria_path = path / "inclusion_criteria.txt"
    criteria = criteria_path.read_text() if criteria_path.exists() else ""
    source = manifest.get("files", {}).get("selected_records") or manifest.get("files", {}).get("labeled_clusters") or manifest.get("files", {}).get("deduplicated_records")
    source_ready = bool(source and (path / source).exists())
    error = None
    fallback_available = credential_available("OPENAI_API_KEY")

    if request.method == "POST" and request.form.get("action") == "review":
        results_name = manifest.get("files", {}).get("ai_screening_full_results")
        if not results_name or not (path / results_name).exists():
            flash("Run AI screening before recording a human decision.")
        else:
            try:
                record_key = validate_text(request.form.get("record_key"), "Record key", 64, required=True)
                decision = validate_text(request.form.get("human_decision"), "Human decision", 20)
                note = validate_text(request.form.get("human_note"), "Review note", 2_000)
                reviewed = save_human_review(
                    path / results_name,
                    path / "human_screening_decisions.csv",
                    path / "human_screening_reviewed_results.csv",
                    record_key=record_key,
                    decision=decision,
                    note=note,
                )
                manifest.setdefault("files", {})["human_screening_decisions"] = "human_screening_decisions.csv"
                manifest["files"]["human_screening_reviewed_results"] = "human_screening_reviewed_results.csv"
                manifest["human_review_rows"] = int(reviewed["final_decision_source"].eq("human").sum())
                manifest["human_review_pending_rows"] = int(reviewed["requires_human_review"].sum())
                manifest["final_screening_decision_counts"] = reviewed["final_decision"].value_counts().to_dict()
                invalidate_extraction(manifest)
                save_manifest(path, manifest)
                flash("Human screening decision saved." if decision else "Human screening decision cleared.")
            except ValueError as exc:
                flash(str(exc))
        return redirect(url_for(
            "screening.screening", project_id=project_id,
            page=request.form.get("page", 1), q=request.form.get("q", ""),
            review_filter=request.form.get("review_filter", "all"),
        ))

    if request.method == "POST":
        if not source_ready:
            error = "Select or deduplicate records before running screening"
        else:
            try:
                criteria = validate_text(
                    criteria, "Inclusion/exclusion criteria",
                    current_app.config["MAX_INCLUSION_CRITERIA_LENGTH"], required=True,
                )
                submitted_key = validate_text(
                    request.form.get("openai_api_key"), "OpenAI API key",
                    current_app.config["MAX_API_KEY_LENGTH"],
                )
                api_key, key_source = resolve_api_key(submitted_key, "OPENAI_API_KEY", required=True)
                model = validate_text(request.form.get("model"), "Screening model", 100, required=True)
                if model not in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}:
                    raise ValueError("Choose a supported OpenAI screening model")
                resume = request.form.get("resume") == "on"

                def run_screening(progress):
                    df = screen_csv(
                        path / source,
                        criteria,
                        path / "ai_screening_full_results.csv",
                        api_key=api_key,
                        model=model,
                        resume=resume,
                        progress_callback=progress,
                    )
                    updated_manifest = load_manifest(path)
                    files = updated_manifest.setdefault("files", {})
                    files["ai_screening_full_results"] = "ai_screening_full_results.csv"
                    for key in ("human_screening_decisions", "human_screening_reviewed_results"):
                        files.pop(key, None)
                    for key in ("human_review_rows", "human_review_pending_rows", "final_screening_decision_counts"):
                        updated_manifest.pop(key, None)
                    invalidate_extraction(updated_manifest)
                    updated_manifest["screening_rows"] = len(df)
                    updated_manifest["screening_decision_counts"] = df["ai_decision"].value_counts().to_dict()
                    updated_manifest["screening_truncated_rows"] = int(df["ai_input_truncated"].fillna(False).sum())
                    updated_manifest["human_review_rows"] = 0
                    updated_manifest["human_review_pending_rows"] = int(
                        (df["ai_decision"].fillna("").eq("uncertain") | df["ai_confidence"].fillna("").eq("low")).sum()
                    )
                    updated_manifest["screening_model"] = model
                    updated_manifest["openai_key_source"] = key_source
                    save_manifest(path, updated_manifest)

                task = current_app.extensions["task_manager"].submit(
                    path,
                    kind="screening",
                    title="Run OpenAI screening",
                    target=run_screening,
                    result_url=url_for("screening.screening", project_id=project_id),
                    failure_message="OpenAI screening failed. Check the key, account access, and record content, then resume the task.",
                )
                return redirect(url_for("screening.screening", project_id=project_id, task=task["task_id"]))
            except ValueError as exc:
                error = str(exc)
        if error:
            return render_template(
                "screening.html", manifest=manifest, criteria=criteria, rows=None, error=error,
                source_ready=source_ready, fallback_available=fallback_available, pagination=None,
                review_stats=None,
            ), 400

    reviewed = _load_reviewed_results(path, manifest)
    rows = None
    pagination = None
    review_stats = None
    if reviewed is not None:
        review_stats = {
            "reviewed": int(reviewed["final_decision_source"].eq("human").sum()),
            "pending": int(reviewed["requires_human_review"].sum()),
            "total": len(reviewed),
        }
        page_df, pagination = _filtered_page(reviewed)
        rows = _preview_rows(page_df)
    return render_template(
        "screening.html", manifest=manifest, criteria=criteria, rows=rows, error=error,
        source_ready=source_ready, fallback_available=fallback_available,
        pagination=pagination, review_stats=review_stats,
    )
