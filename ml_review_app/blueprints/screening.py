"""OpenAI screening and staged human abstract/full-text review routes."""

from __future__ import annotations

import math
import re
from urllib.parse import quote

import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..services.credential_service import credential_available, resolve_api_key
from ..services.full_text_service import apply_full_text_reviews, full_text_path, save_full_text_review
from ..services.project_service import invalidate_extraction, load_manifest, project_dir, save_manifest
from ..services.screening_service import (
    EXCLUSION_CATEGORY_LABELS,
    apply_human_reviews,
    save_human_review,
    screen_csv,
)
from ..services.validation_service import validate_text

bp = Blueprint("screening", __name__, url_prefix="/projects/<project_id>/screening")
LIST_PAGE_SIZE = 30
REVIEW_FILTERS = {"all", "pending", "priority", "reviewed", "include", "exclude", "uncertain"}


def _identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def _source_urls(row: dict) -> tuple[str, str]:
    pmid = _identifier(row.get("PMID"))
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if re.fullmatch(r"\d+", pmid) else ""
    doi = _identifier(row.get("DOI"))
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE).strip()
    doi_url = f"https://doi.org/{quote(doi, safe='/')}" if doi else ""
    return pubmed_url, doi_url


def _preview_rows(df: pd.DataFrame, project_path, manifest: dict) -> list[dict]:
    columns = [
        "record_key", "RecordID", "PMID", "DOI", "Title", "Abstract", "Authors", "Journal", "Date", "Year",
        "ai_decision", "ai_confidence", "ai_exclusion_category", "ai_exclusion_reason",
        "ai_population_match", "ai_exposure_match", "ai_outcome_match", "ai_study_design_appropriate",
        "ai_reasoning", "ai_input_truncated", "human_decision", "human_note", "human_reviewed_at",
        "abstract_final_decision", "abstract_final_decision_source", "full_text_decision",
        "full_text_exclusion_category", "full_text_exclusion_reason", "full_text_note", "full_text_reviewed_at",
        "final_decision", "final_decision_source", "requires_human_review", "requires_full_text_review",
    ]
    preview = df[[column for column in columns if column in df.columns]].copy()
    for column in (
        "ai_population_match", "ai_exposure_match", "ai_outcome_match", "ai_study_design_appropriate",
        "ai_input_truncated", "requires_human_review", "requires_full_text_review",
    ):
        if column in preview:
            preview[column] = preview[column].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    rows = preview.fillna("").to_dict(orient="records")
    documents = manifest.get("full_text_documents", {})
    for row in rows:
        row["PMID"] = _identifier(row.get("PMID"))
        row["RecordID"] = _identifier(row.get("RecordID"))
        row["pubmed_url"], row["doi_url"] = _source_urls(row)
        record_key = str(row.get("record_key", ""))
        metadata = documents.get(record_key)
        row["full_text"] = metadata if metadata and full_text_path(project_path, manifest, record_key) else None
    return rows


def _load_abstract_results(path, manifest: dict) -> pd.DataFrame | None:
    results_name = manifest.get("files", {}).get("ai_screening_full_results")
    if not results_name or not (path / results_name).exists():
        return None
    screening_df = pd.read_csv(path / results_name)
    reviews_name = manifest.get("files", {}).get("human_screening_decisions")
    reviews_df = pd.read_csv(path / reviews_name, dtype=str) if reviews_name and (path / reviews_name).exists() else None
    return apply_human_reviews(screening_df, reviews_df)


def _load_full_text_results(path, manifest: dict, abstract: pd.DataFrame) -> pd.DataFrame:
    reviews_name = manifest.get("files", {}).get("human_full_text_decisions")
    reviews_df = pd.read_csv(path / reviews_name, dtype=str) if reviews_name and (path / reviews_name).exists() else None
    return apply_full_text_reviews(abstract, reviews_df)


def _stage_stats(abstract: pd.DataFrame, full_text: pd.DataFrame) -> dict[str, dict[str, int]]:
    abstract_reviewed = abstract["human_decision"].fillna("").isin({"include", "exclude", "uncertain"})
    full_eligible = full_text["full_text_eligible"].fillna(False).astype(bool)
    full_reviewed = full_text["full_text_decision"].fillna("").isin({"include", "exclude", "uncertain"}) & full_eligible
    return {
        "abstract": {
            "total": len(abstract),
            "reviewed": int(abstract_reviewed.sum()),
            "pending": int((~abstract_reviewed).sum()),
            "priority": int(abstract["requires_human_review"].fillna(False).sum()),
        },
        "full_text": {
            "total": int(full_eligible.sum()),
            "reviewed": int(full_reviewed.sum()),
            "pending": int((full_eligible & ~full_reviewed).sum()),
            "with_pdf": 0,
        },
    }


def _workspace(df: pd.DataFrame, *, stage: str) -> tuple[pd.DataFrame, dict]:
    review_filter = request.args.get("review_filter", "all")
    if review_filter == "needs_review":
        review_filter = "priority"
    if review_filter not in REVIEW_FILTERS:
        review_filter = "all"
    query = request.args.get("q", "").strip()[:300]
    view = request.args.get("view", "focus")
    if view not in {"focus", "list"}:
        view = "focus"

    filtered = df.copy()
    if stage == "full_text":
        filtered = filtered.loc[filtered["full_text_eligible"].fillna(False)]
        reviewed = filtered["full_text_decision"].fillna("").isin({"include", "exclude", "uncertain"})
        pending = filtered["requires_full_text_review"].fillna(False)
    else:
        reviewed = filtered["human_decision"].fillna("").isin({"include", "exclude", "uncertain"})
        pending = ~reviewed
    if review_filter == "pending":
        filtered = filtered.loc[pending]
    elif review_filter == "priority" and stage == "abstract":
        filtered = filtered.loc[filtered["requires_human_review"].fillna(False)]
    elif review_filter == "reviewed":
        filtered = filtered.loc[reviewed]
    elif review_filter in {"include", "exclude", "uncertain"}:
        decision_column = "full_text_decision" if stage == "full_text" else "final_decision"
        filtered = filtered.loc[filtered[decision_column].fillna("").eq(review_filter)]
    if query:
        searchable = filtered.reindex(columns=[
            "RecordID", "PMID", "DOI", "Title", "Abstract", "Authors", "Journal", "ai_decision",
            "ai_exclusion_reason", "ai_reasoning", "human_note", "full_text_exclusion_reason", "full_text_note",
        ]).fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered.loc[searchable.str.contains(query, case=False, regex=False)]
    filtered = filtered.reset_index(drop=True)
    total = len(filtered)

    if view == "focus":
        requested_key = request.args.get("record", "")
        keys = filtered.get("record_key", pd.Series(dtype="object")).astype(str).tolist()
        if requested_key in keys:
            position = keys.index(requested_key)
        else:
            if stage == "full_text":
                pending_positions = filtered.index[filtered["requires_full_text_review"].fillna(False)].tolist()
            else:
                pending_positions = filtered.index[~filtered["human_decision"].fillna("").isin({"include", "exclude", "uncertain"})].tolist()
            position = pending_positions[0] if pending_positions else 0
        page_df = filtered.iloc[[position]] if total else filtered
        return page_df, {
            "stage": stage, "view": view, "q": query, "review_filter": review_filter,
            "filtered_count": total, "position": position + 1 if total else 0,
            "previous_record": keys[position - 1] if total and position > 0 else "",
            "next_record": keys[position + 1] if total and position + 1 < total else "",
            "page": 1, "pages": 1, "start": position + 1 if total else 0, "end": position + 1 if total else 0,
        }

    pages = max(1, math.ceil(total / LIST_PAGE_SIZE))
    try:
        page = max(1, min(int(request.args.get("page", 1)), pages))
    except ValueError:
        page = 1
    start = (page - 1) * LIST_PAGE_SIZE
    return filtered.iloc[start:start + LIST_PAGE_SIZE], {
        "stage": stage, "view": view, "q": query, "review_filter": review_filter,
        "filtered_count": total, "page": page, "pages": pages, "position": 0,
        "previous_record": "", "next_record": "",
        "start": start + 1 if total else 0, "end": min(start + LIST_PAGE_SIZE, total),
    }


def _review_redirect(project_id: str):
    parameters = {
        "stage": request.form.get("stage", "abstract"),
        "view": request.form.get("view", "focus"),
        "review_filter": request.form.get("review_filter", "all"),
        "q": request.form.get("q", ""),
        "page": request.form.get("page", "1"),
    }
    next_record = request.form.get("next_record", "")
    current_record = request.form.get("record_key", "")
    if parameters["view"] == "focus":
        parameters["record"] = next_record or current_record
    return redirect(url_for("screening.screening", project_id=project_id, **parameters))


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
    action = request.form.get("action", "")

    if request.method == "POST" and action in {"review", "abstract_review"}:
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
                files = manifest.setdefault("files", {})
                files["human_screening_decisions"] = "human_screening_decisions.csv"
                files["human_screening_reviewed_results"] = "human_screening_reviewed_results.csv"
                human_reviewed = reviewed["final_decision_source"].eq("human")
                manifest["human_review_rows"] = int(human_reviewed.sum())
                manifest["abstract_review_pending_rows"] = int((~human_reviewed).sum())
                manifest["human_review_pending_rows"] = int(reviewed["requires_human_review"].sum())
                manifest["final_screening_decision_counts"] = reviewed["final_decision"].value_counts().to_dict()
                full_reviews_name = files.get("human_full_text_decisions")
                if full_reviews_name and (path / full_reviews_name).is_file():
                    full_reviews = pd.read_csv(path / full_reviews_name, dtype=str)
                else:
                    full_reviews = None
                final_results = apply_full_text_reviews(reviewed, full_reviews)
                manifest["full_text_review_rows"] = int(final_results["final_decision_source"].eq("human_full_text").sum())
                manifest["full_text_review_pending_rows"] = int(final_results["requires_full_text_review"].sum())
                manifest["full_text_screening_decision_counts"] = final_results["final_decision"].value_counts().to_dict()
                if full_reviews is not None:
                    final_results.to_csv(path / "full_text_screening_results.csv", index=False)
                    files["full_text_screening_results"] = "full_text_screening_results.csv"
                    manifest["final_screening_decision_counts"] = final_results["final_decision"].value_counts().to_dict()
                invalidate_extraction(manifest)
                save_manifest(path, manifest)
                flash("Title and abstract decision saved." if decision else "Title and abstract decision cleared.")
            except ValueError as exc:
                flash(str(exc))
        return _review_redirect(project_id)

    if request.method == "POST" and action == "full_text_review":
        files = manifest.get("files", {})
        abstract_name = files.get("human_screening_reviewed_results") or files.get("ai_screening_full_results")
        if not abstract_name or not (path / abstract_name).is_file():
            flash("Complete title and abstract screening before full-text review.")
        else:
            try:
                reviewed = save_full_text_review(
                    path / abstract_name,
                    path / "human_full_text_decisions.csv",
                    path / "full_text_screening_results.csv",
                    record_key=validate_text(request.form.get("record_key"), "Record key", 64, required=True),
                    decision=validate_text(request.form.get("full_text_decision"), "Full-text decision", 20),
                    exclusion_category=validate_text(request.form.get("exclusion_category"), "Exclusion category", 100),
                    exclusion_reason=validate_text(request.form.get("exclusion_reason"), "Exclusion reason", 400),
                    note=validate_text(request.form.get("full_text_note"), "Full-text review note", 2_000),
                )
                manifest.setdefault("files", {})["human_full_text_decisions"] = "human_full_text_decisions.csv"
                manifest["files"]["full_text_screening_results"] = "full_text_screening_results.csv"
                full_reviewed = reviewed["final_decision_source"].eq("human_full_text")
                manifest["full_text_review_rows"] = int(full_reviewed.sum())
                manifest["full_text_review_pending_rows"] = int(reviewed["requires_full_text_review"].sum())
                manifest["final_screening_decision_counts"] = reviewed["final_decision"].value_counts().to_dict()
                manifest["full_text_screening_decision_counts"] = manifest["final_screening_decision_counts"]
                invalidate_extraction(manifest)
                save_manifest(path, manifest)
                decision = request.form.get("full_text_decision", "")
                flash("Full-text decision saved." if decision else "Full-text decision cleared.")
            except ValueError as exc:
                flash(str(exc))
        return _review_redirect(project_id)

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
                    for key in (
                        "human_screening_decisions", "human_screening_reviewed_results",
                        "human_full_text_decisions", "full_text_screening_results",
                    ):
                        files.pop(key, None)
                    for key in (
                        "human_review_rows", "human_review_pending_rows", "abstract_review_pending_rows",
                        "full_text_review_rows", "full_text_review_pending_rows", "final_screening_decision_counts",
                    ):
                        updated_manifest.pop(key, None)
                    invalidate_extraction(updated_manifest)
                    updated_manifest["screening_rows"] = len(df)
                    updated_manifest["screening_decision_counts"] = df["ai_decision"].value_counts().to_dict()
                    updated_manifest["screening_truncated_rows"] = int(df["ai_input_truncated"].fillna(False).sum())
                    updated_manifest["human_review_rows"] = 0
                    updated_manifest["abstract_review_pending_rows"] = len(df)
                    updated_manifest["human_review_pending_rows"] = int(
                        (df["ai_decision"].fillna("").eq("uncertain") | df["ai_confidence"].fillna("").eq("low")).sum()
                    )
                    initial_full_text = apply_full_text_reviews(apply_human_reviews(df))
                    updated_manifest["full_text_review_rows"] = 0
                    updated_manifest["full_text_review_pending_rows"] = int(
                        initial_full_text["requires_full_text_review"].sum()
                    )
                    updated_manifest["full_text_screening_decision_counts"] = (
                        initial_full_text["final_decision"].value_counts().to_dict()
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
                stage="abstract", stage_stats=None, exclusion_categories=EXCLUSION_CATEGORY_LABELS,
            ), 400

    abstract = _load_abstract_results(path, manifest)
    rows = None
    pagination = None
    stats = None
    stage = request.args.get("stage", "abstract")
    if stage not in {"abstract", "full_text"}:
        stage = "abstract"
    if abstract is not None:
        full_text = _load_full_text_results(path, manifest, abstract)
        stats = _stage_stats(abstract, full_text)
        eligible_keys = set(full_text.loc[full_text["full_text_eligible"], "record_key"].astype(str))
        stats["full_text"]["with_pdf"] = sum(
            1 for key in manifest.get("full_text_documents", {})
            if key in eligible_keys and full_text_path(path, manifest, key)
        )
        page_df, pagination = _workspace(full_text if stage == "full_text" else abstract, stage=stage)
        rows = _preview_rows(page_df, path, manifest)
    return render_template(
        "screening.html", manifest=manifest, criteria=criteria, rows=rows, error=error,
        source_ready=source_ready, fallback_available=fallback_available, pagination=pagination,
        stage=stage, stage_stats=stats, exclusion_categories=EXCLUSION_CATEGORY_LABELS,
    )
