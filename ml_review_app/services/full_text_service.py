"""Project-scoped full-text PDF storage and human full-text adjudication."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from openai import OpenAI
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .screening_service import EXCLUSION_CATEGORY_LABELS, ScreeningDecision, apply_human_reviews

FULL_TEXT_DIRECTORY = "full_texts"
MAX_FULL_TEXT_PDF_BYTES = 50 * 1024 * 1024
FULL_TEXT_AI_SCHEMA_VERSION = 1
FULL_TEXT_REVIEW_COLUMNS = [
    "record_key",
    "full_text_decision",
    "full_text_exclusion_category",
    "full_text_exclusion_reason",
    "full_text_note",
    "full_text_reviewed_at",
    "full_text_review_status",
    "full_text_auto_reviewed_at",
    "full_text_auto_reviewed_ai_decision",
    "full_text_auto_reviewed_config_hash",
]
FULL_TEXT_AI_COLUMNS = [
    *[f"full_text_ai_{name}" for name in ScreeningDecision.model_fields],
    "full_text_ai_model",
    "full_text_ai_config_hash",
    "full_text_ai_screened_at",
    "full_text_ai_filename",
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


def _full_text_ai_config_hash(criteria: str, model: str, pdf: Path) -> str:
    source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    return hashlib.sha256(
        f"schema={FULL_TEXT_AI_SCHEMA_VERSION}\nmodel={model}\npdf={source_hash}\n{criteria}".encode()
    ).hexdigest()


def screen_full_text_record(
    title: str,
    criteria: str,
    pdf: Path,
    *,
    api_key: str,
    model: str,
    record_identifier: str,
    client: Any | None = None,
) -> ScreeningDecision:
    """Screen one uploaded full-text PDF through OpenAI Structured Outputs."""

    if not api_key:
        raise ValueError("An OpenAI API key is required")
    encoded_pdf = base64.b64encode(pdf.read_bytes()).decode("ascii")
    api_client = client or OpenAI(api_key=api_key)
    response = api_client.responses.parse(
        model=model,
        instructions=(
            "You are assisting a systematic-review team with full-text eligibility screening. "
            "Apply the supplied criteria to the complete uploaded article. Choose uncertain when the full text "
            "does not establish eligibility. For an excluded record, select exactly one broad exclusion category "
            "and give a concise record-specific reason. For included or uncertain records, leave exclusion fields "
            "null. Do not invent study details. AI output is decision support; a human decision always overrides it.\n\n"
            f"INCLUSION AND EXCLUSION CRITERIA:\n{criteria}"
        ),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"TITLE:\n{title}"},
                {
                    "type": "input_file",
                    "filename": pdf.name,
                    "file_data": f"data:application/pdf;base64,{encoded_pdf}",
                },
            ],
        }],
        text_format=ScreeningDecision,
        reasoning={"effort": "low"},
        safety_identifier="ml-review-full-text-" + hashlib.sha256(record_identifier.encode()).hexdigest()[:24],
        store=False,
    )
    if response.output_parsed is None:
        raise ValueError("OpenAI did not return a full-text screening decision")
    return response.output_parsed


def screen_full_text_csv(
    input_csv: Path,
    criteria: str,
    output_csv: Path,
    *,
    full_text_files: dict[str, Path],
    api_key: str,
    model: str,
    resume: bool = True,
    client: Any | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Screen eligible uploaded PDFs and save resumable per-record AI decisions."""

    source = pd.read_csv(input_csv)
    if "record_key" not in source:
        source = apply_human_reviews(source)
    source = source.reset_index(drop=True)
    abstract_decisions = source.get("final_decision", source["ai_decision"]).fillna("").astype(str).str.lower()
    candidate_indices = [
        index for index, row in source.iterrows()
        if abstract_decisions.loc[index] in {"include", "uncertain"}
        and str(row["record_key"]) in full_text_files
        and full_text_files[str(row["record_key"])].is_file()
    ]
    if not candidate_indices:
        raise ValueError("Upload at least one PDF for an eligible record before running full-text AI screening")

    result = source.copy()
    for column in FULL_TEXT_AI_COLUMNS:
        result[column] = None
    expected_configs = {
        str(source.loc[index, "record_key"]): _full_text_ai_config_hash(
            criteria, model, full_text_files[str(source.loc[index, "record_key"])]
        )
        for index in candidate_indices
    }
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv, dtype={"record_key": str}).fillna("")
        if "record_key" in existing:
            existing = existing.drop_duplicates("record_key", keep="last").set_index("record_key")
            for index in candidate_indices:
                record_key = str(source.loc[index, "record_key"])
                if record_key not in existing.index:
                    continue
                previous = existing.loc[record_key]
                if str(previous.get("full_text_ai_config_hash", "")) != expected_configs[record_key]:
                    continue
                for column in FULL_TEXT_AI_COLUMNS:
                    if column in previous:
                        result.loc[index, column] = previous[column]

    completed = int(result.loc[candidate_indices, "full_text_ai_decision"].fillna("").ne("").sum())
    total = len(candidate_indices)
    if progress_callback:
        progress_callback(completed, total, "Resuming saved full-text decisions" if completed else "Preparing full-text screening")
    for index in candidate_indices:
        if str(result.loc[index, "full_text_ai_decision"] or ""):
            continue
        row = source.loc[index]
        record_key = str(row["record_key"])
        pdf = full_text_files[record_key]
        title = "" if pd.isna(row.get("Title")) else str(row.get("Title", ""))
        decision = screen_full_text_record(
            title,
            criteria,
            pdf,
            api_key=api_key,
            model=model,
            record_identifier=record_key,
            client=client,
        )
        for name, value in decision.model_dump().items():
            result.loc[index, f"full_text_ai_{name}"] = value
        result.loc[index, "full_text_ai_model"] = model
        result.loc[index, "full_text_ai_config_hash"] = expected_configs[record_key]
        result.loc[index, "full_text_ai_screened_at"] = datetime.now(timezone.utc).isoformat()
        result.loc[index, "full_text_ai_filename"] = pdf.name
        result.to_csv(output_csv, index=False)
        completed += 1
        if progress_callback:
            progress_callback(completed, total, f"Screened {completed} of {total} full texts")
    result.to_csv(output_csv, index=False)
    if progress_callback:
        progress_callback(total, total, "Full-text AI screening CSV saved")
    return result, total


def apply_full_text_reviews(
    abstract_reviewed_df: pd.DataFrame,
    reviews_df: pd.DataFrame | None = None,
    ai_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Overlay full-text AI and human decisions while preserving every audit layer."""

    result = abstract_reviewed_df.reset_index(drop=True).copy()
    if "record_key" not in result:
        result = apply_human_reviews(result)
    ai_lookup: dict[str, dict[str, Any]] = {}
    if ai_results_df is not None and not ai_results_df.empty and "record_key" in ai_results_df:
        ai_lookup = ai_results_df.drop_duplicates("record_key", keep="last").set_index("record_key").to_dict(orient="index")
    for column in FULL_TEXT_AI_COLUMNS:
        result[column] = result["record_key"].map(lambda key: ai_lookup.get(key, {}).get(column))
    review_lookup: dict[str, dict[str, Any]] = {}
    if reviews_df is not None and not reviews_df.empty and "record_key" in reviews_df:
        review_lookup = reviews_df.drop_duplicates("record_key", keep="last").set_index("record_key").to_dict(orient="index")
    for column in FULL_TEXT_REVIEW_COLUMNS[1:]:
        result[column] = result["record_key"].map(lambda key: review_lookup.get(key, {}).get(column))

    abstract_decision = result.get("final_decision", result["ai_decision"]).fillna("").astype(str).str.lower()
    abstract_source = result.get("final_decision_source", pd.Series("ai", index=result.index))
    eligible = abstract_decision.isin({"include", "uncertain"})
    ai_reviewed = result["full_text_ai_decision"].fillna("").isin({"include", "exclude", "uncertain"}) & eligible
    reviewed = result["full_text_decision"].fillna("").isin({"include", "exclude", "uncertain"}) & eligible
    auto_reviewed = (
        result["full_text_review_status"].fillna("").eq("ai_accepted")
        & result["full_text_auto_reviewed_ai_decision"].fillna("").eq(result["full_text_ai_decision"].fillna(""))
        & result["full_text_auto_reviewed_config_hash"].fillna("").eq(result["full_text_ai_config_hash"].fillna(""))
        & ai_reviewed & ~reviewed
    )
    ai_or_abstract_decision = result["full_text_ai_decision"].where(ai_reviewed, abstract_decision)
    ai_or_abstract_source = abstract_source.where(~ai_reviewed, "full_text_ai")
    result["abstract_final_decision"] = abstract_decision
    result["abstract_final_decision_source"] = abstract_source
    result["final_decision"] = result["full_text_decision"].where(reviewed, ai_or_abstract_decision)
    result["final_decision_source"] = ai_or_abstract_source
    result.loc[auto_reviewed, "final_decision_source"] = "full_text_ai_auto_reviewed"
    result.loc[reviewed, "final_decision_source"] = "human_full_text"
    result["full_text_eligible"] = eligible
    result["full_text_ai_available"] = ai_reviewed
    result["full_text_review_complete"] = reviewed | auto_reviewed
    result["full_text_ai_human_disagreement"] = (
        reviewed & ai_reviewed
        & result["full_text_decision"].fillna("").ne(result["full_text_ai_decision"].fillna(""))
    )
    result["requires_full_text_review"] = eligible & ~result["full_text_review_complete"]
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
    ai_results_csv: Path | None = None,
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
    ai_results = pd.read_csv(ai_results_csv) if ai_results_csv and ai_results_csv.is_file() else None
    keyed = apply_full_text_reviews(abstract, ai_results_df=ai_results)
    available = keyed.loc[keyed["full_text_eligible"], "record_key"]
    if record_key not in set(available):
        raise ValueError("This record is not eligible for full-text review")
    if reviews_csv.exists():
        reviews = pd.read_csv(reviews_csv, dtype=str).fillna("")
        reviews = reviews.reindex(columns=FULL_TEXT_REVIEW_COLUMNS, fill_value="")
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
            "full_text_review_status": "human_reviewed",
            "full_text_auto_reviewed_at": "",
            "full_text_auto_reviewed_ai_decision": "",
            "full_text_auto_reviewed_config_hash": "",
        }
    reviews.to_csv(reviews_csv, index=False)
    reviewed = apply_full_text_reviews(abstract, reviews, ai_results)
    reviewed.to_csv(reviewed_results_csv, index=False)
    return reviewed


def mark_full_text_ai_accepted(
    abstract_results_csv: Path,
    reviews_csv: Path,
    reviewed_results_csv: Path,
    ai_results_csv: Path,
    *,
    record_keys: set[str],
) -> tuple[pd.DataFrame, int]:
    """Mark selected full-text AI decisions accepted without creating human decisions."""

    abstract = pd.read_csv(abstract_results_csv)
    if "record_key" not in abstract:
        abstract = apply_human_reviews(abstract)
    ai_results = pd.read_csv(ai_results_csv)
    if reviews_csv.exists():
        reviews = pd.read_csv(reviews_csv, dtype=str).fillna("")
        reviews = reviews.reindex(columns=FULL_TEXT_REVIEW_COLUMNS, fill_value="")
    else:
        reviews = pd.DataFrame(columns=FULL_TEXT_REVIEW_COLUMNS)
    current = apply_full_text_reviews(abstract, reviews, ai_results)
    available = set(current.loc[current["full_text_ai_available"] & ~current["full_text_review_complete"], "record_key"])
    selected = record_keys & available
    if not selected:
        raise ValueError("No matching unreviewed full-text AI decisions are available to accept")
    reviews = reviews.loc[~reviews["record_key"].isin(selected)].copy()
    reviewed_at = datetime.now(timezone.utc).isoformat()
    current_by_key = current.set_index("record_key")
    additions = pd.DataFrame([
        {
            "record_key": record_key,
            "full_text_decision": "",
            "full_text_exclusion_category": "",
            "full_text_exclusion_reason": "",
            "full_text_note": "",
            "full_text_reviewed_at": "",
            "full_text_review_status": "ai_accepted",
            "full_text_auto_reviewed_at": reviewed_at,
            "full_text_auto_reviewed_ai_decision": current_by_key.loc[record_key, "full_text_ai_decision"],
            "full_text_auto_reviewed_config_hash": current_by_key.loc[record_key, "full_text_ai_config_hash"],
        }
        for record_key in sorted(selected)
    ])
    reviews = pd.concat([reviews, additions], ignore_index=True)
    reviews.to_csv(reviews_csv, index=False)
    reviewed = apply_full_text_reviews(abstract, reviews, ai_results)
    reviewed.to_csv(reviewed_results_csv, index=False)
    return reviewed, len(selected)
