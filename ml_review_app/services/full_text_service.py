"""Project-scoped full-text PDF storage and human full-text adjudication."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .screening_service import EXCLUSION_CATEGORY_LABELS, apply_human_reviews

FULL_TEXT_DIRECTORY = "full_texts"
MAX_FULL_TEXT_PDF_BYTES = 50 * 1024 * 1024
FULL_TEXT_REVIEW_COLUMNS = [
    "record_key",
    "full_text_decision",
    "full_text_exclusion_category",
    "full_text_exclusion_reason",
    "full_text_note",
    "full_text_reviewed_at",
]


def validate_record_key(record_key: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{24}", record_key):
        raise ValueError("The selected record is invalid")
    return record_key


def full_text_path(project_path: Path, manifest: dict[str, Any], record_key: str) -> Path | None:
    """Resolve a manifest-backed full-text path without allowing traversal."""

    validate_record_key(record_key)
    record = manifest.get("full_text_documents", {}).get(record_key)
    filename = record.get("filename") if isinstance(record, dict) else None
    if filename != f"{FULL_TEXT_DIRECTORY}/{record_key}.pdf":
        return None
    candidate = (project_path / filename).resolve()
    directory = project_path / FULL_TEXT_DIRECTORY
    expected_root = directory.resolve()
    if (
        directory.is_symlink()
        or expected_root.parent != project_path.resolve()
        or candidate.parent != expected_root
        or not candidate.is_file()
    ):
        return None
    return candidate


def save_full_text_pdf(
    project_path: Path,
    manifest: dict[str, Any],
    record_key: str,
    upload: FileStorage | None,
    *,
    valid_record_keys: set[str],
) -> dict[str, Any]:
    """Validate and atomically save one PDF, returning its manifest metadata."""

    record_key = validate_record_key(record_key)
    if record_key not in valid_record_keys:
        raise ValueError("The selected record is no longer available")
    if upload is None or not upload.filename:
        raise ValueError("Choose a PDF to upload")
    if not upload.filename.lower().endswith(".pdf"):
        raise ValueError("Full text must be uploaded as a PDF")
    data = upload.stream.read(MAX_FULL_TEXT_PDF_BYTES + 1)
    if len(data) > MAX_FULL_TEXT_PDF_BYTES:
        raise ValueError("Full-text PDF must be 50 MB or smaller")
    if b"%PDF-" not in data[:1024]:
        raise ValueError("The uploaded file does not appear to be a valid PDF")

    directory = project_path / FULL_TEXT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or directory.resolve().parent != project_path.resolve():
        raise ValueError("Project full-text storage is unsafe")
    destination = directory / f"{record_key}.pdf"
    temporary = directory / f".{record_key}.pending.pdf"
    temporary.write_bytes(data)
    temporary.replace(destination)
    original_name = secure_filename(upload.filename)[:240] or "full-text.pdf"
    return {
        "filename": f"{FULL_TEXT_DIRECTORY}/{record_key}.pdf",
        "original_filename": original_name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def remove_full_text_pdf(project_path: Path, manifest: dict[str, Any], record_key: str) -> None:
    """Remove one managed full-text PDF if present."""

    path = full_text_path(project_path, manifest, record_key)
    if path is not None:
        path.unlink()


def apply_full_text_reviews(
    abstract_reviewed_df: pd.DataFrame,
    reviews_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Overlay full-text decisions while preserving the abstract-stage audit trail."""

    result = abstract_reviewed_df.reset_index(drop=True).copy()
    if "record_key" not in result:
        result = apply_human_reviews(result)
    review_lookup: dict[str, dict[str, Any]] = {}
    if reviews_df is not None and not reviews_df.empty and "record_key" in reviews_df:
        review_lookup = reviews_df.drop_duplicates("record_key", keep="last").set_index("record_key").to_dict(orient="index")
    for column in FULL_TEXT_REVIEW_COLUMNS[1:]:
        result[column] = result["record_key"].map(lambda key: review_lookup.get(key, {}).get(column))

    abstract_decision = result.get("final_decision", result["ai_decision"]).fillna("").astype(str).str.lower()
    abstract_source = result.get("final_decision_source", pd.Series("ai", index=result.index))
    eligible = abstract_decision.isin({"include", "uncertain"})
    reviewed = result["full_text_decision"].fillna("").isin({"include", "exclude", "uncertain"}) & eligible
    result["abstract_final_decision"] = abstract_decision
    result["abstract_final_decision_source"] = abstract_source
    result["final_decision"] = result["full_text_decision"].where(reviewed, abstract_decision)
    result["final_decision_source"] = abstract_source.where(~reviewed, "human_full_text")
    result["full_text_eligible"] = eligible
    result["requires_full_text_review"] = eligible & ~reviewed
    return result


def save_full_text_review(
    abstract_results_csv: Path,
    reviews_csv: Path,
    reviewed_results_csv: Path,
    *,
    record_key: str,
    decision: str,
    exclusion_category: str,
    exclusion_reason: str,
    note: str,
) -> pd.DataFrame:
    """Upsert or clear one full-text decision and rebuild the final eligibility export."""

    record_key = validate_record_key(record_key)
    if decision not in {"", "include", "exclude", "uncertain"}:
        raise ValueError("Choose include, exclude, uncertain, or clear the review")
    if decision == "exclude":
        if exclusion_category not in EXCLUSION_CATEGORY_LABELS:
            raise ValueError("Choose a broad exclusion category")
        if not exclusion_reason.strip():
            raise ValueError("Enter a concise full-text exclusion reason")
    else:
        exclusion_category = ""
        exclusion_reason = ""

    abstract = pd.read_csv(abstract_results_csv)
    if "record_key" not in abstract:
        abstract = apply_human_reviews(abstract)
    keyed = apply_full_text_reviews(abstract)
    available = keyed.loc[keyed["full_text_eligible"], "record_key"]
    if record_key not in set(available):
        raise ValueError("This record is not eligible for full-text review")
    if reviews_csv.exists():
        reviews = pd.read_csv(reviews_csv, dtype=str).fillna("")
    else:
        reviews = pd.DataFrame(columns=FULL_TEXT_REVIEW_COLUMNS)
    reviews = reviews.loc[reviews["record_key"] != record_key].copy()
    if decision:
        reviews.loc[len(reviews)] = {
            "record_key": record_key,
            "full_text_decision": decision,
            "full_text_exclusion_category": exclusion_category,
            "full_text_exclusion_reason": exclusion_reason.strip(),
            "full_text_note": note.strip(),
            "full_text_reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    reviews.to_csv(reviews_csv, index=False)
    reviewed = apply_full_text_reviews(abstract, reviews)
    reviewed.to_csv(reviewed_results_csv, index=False)
    return reviewed
