"""Structured OpenAI screening with incremental, resumable outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

MAX_SCREENING_RECORD_LENGTH = 100_000
SCREENING_SCHEMA_VERSION = 2
HUMAN_REVIEW_COLUMNS = [
    "record_key",
    "human_decision",
    "human_note",
    "human_reviewed_at",
    "abstract_review_status",
    "abstract_auto_reviewed_at",
]
ExclusionCategory = Literal[
    "population",
    "exposure",
    "outcome",
    "study_design",
    "publication_type",
    "duplicate",
    "insufficient_information",
    "other",
]
EXCLUSION_CATEGORY_LABELS = {
    "population": "Population",
    "exposure": "Exposure",
    "outcome": "Outcome",
    "study_design": "Study design",
    "publication_type": "Publication type",
    "duplicate": "Duplicate",
    "insufficient_information": "Insufficient information",
    "other": "Other / not specified",
}


class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude", "uncertain"]
    confidence: Literal["high", "medium", "low"]
    exclusion_category: ExclusionCategory | None = Field(
        default=None,
        description="Required for excluded records; null for included or uncertain records",
    )
    exclusion_reason: str | None = Field(
        default=None,
        max_length=400,
        description="Brief record-specific exclusion rationale; null unless the decision is exclude",
    )
    reasoning: str = Field(description="Concise rationale grounded in the supplied title, abstract, and criteria")
    population_match: bool
    exposure_match: bool
    outcome_match: bool
    study_design_appropriate: bool

    @model_validator(mode="after")
    def validate_exclusion_fields(self) -> ScreeningDecision:
        if self.decision == "exclude" and self.exclusion_category is None:
            raise ValueError("Excluded records require an exclusion category")
        if self.decision == "exclude" and not (self.exclusion_reason or "").strip():
            raise ValueError("Excluded records require a concise exclusion reason")
        if self.decision != "exclude" and (self.exclusion_category is not None or self.exclusion_reason is not None):
            raise ValueError("Exclusion category and reason must be null unless the record is excluded")
        return self


def _configuration_hash(criteria: str, model: str) -> str:
    return hashlib.sha256(
        (
            f"{model}\nscreening-schema={SCREENING_SCHEMA_VERSION}"
            f"\nleading-character-limit={MAX_SCREENING_RECORD_LENGTH}\n{criteria}"
        ).encode()
    ).hexdigest()


def _record_identifier(row: pd.Series, index: int) -> str:
    value = row.get("RecordID", index)
    return str(index if pd.isna(value) else value)


def screening_record_key(row: pd.Series, index: int) -> str:
    """Return a stable, non-identifying key for a screening row."""

    identity = {
        "record_id": "" if pd.isna(row.get("RecordID")) else str(row.get("RecordID", "")),
        "pmid": "" if pd.isna(row.get("PMID")) else str(row.get("PMID", "")),
        "doi": "" if pd.isna(row.get("DOI")) else str(row.get("DOI", "")),
        "title": "" if pd.isna(row.get("Title")) else str(row.get("Title", "")),
        "row": index,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def apply_human_reviews(screening_df: pd.DataFrame, reviews_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Overlay human decisions while retaining the complete AI audit trail."""

    result = screening_df.reset_index(drop=True).copy()
    result["record_key"] = [screening_record_key(row, index) for index, row in result.iterrows()]
    review_lookup: dict[str, dict[str, Any]] = {}
    if reviews_df is not None and not reviews_df.empty and "record_key" in reviews_df:
        review_lookup = reviews_df.drop_duplicates("record_key", keep="last").set_index("record_key").to_dict(orient="index")
    for column in HUMAN_REVIEW_COLUMNS[1:]:
        result[column] = result["record_key"].map(lambda key: review_lookup.get(key, {}).get(column))
    reviewed = result["human_decision"].fillna("").isin({"include", "exclude", "uncertain"})
    auto_reviewed = result["abstract_review_status"].fillna("").eq("ai_accepted") & ~reviewed
    result["final_decision"] = result["human_decision"].where(reviewed, result["ai_decision"])
    result["final_decision_source"] = "ai"
    result.loc[auto_reviewed, "final_decision_source"] = "ai_auto_reviewed"
    result.loc[reviewed, "final_decision_source"] = "human"
    result["abstract_review_complete"] = reviewed | auto_reviewed
    result["abstract_ai_human_disagreement"] = (
        reviewed & result["human_decision"].fillna("").ne(result["ai_decision"].fillna(""))
    )
    result["requires_human_review"] = (
        result["ai_decision"].fillna("").eq("uncertain")
        | result["ai_confidence"].fillna("").eq("low")
    ) & ~result["abstract_review_complete"]
    return result


def save_human_review(
    screening_csv: Path,
    reviews_csv: Path,
    reviewed_results_csv: Path,
    *,
    record_key: str,
    decision: str,
    note: str,
) -> pd.DataFrame:
    """Upsert or clear one human decision and rebuild the reviewed export."""

    if decision not in {"", "include", "exclude", "uncertain"}:
        raise ValueError("Choose include, exclude, uncertain, or clear the review")
    screening = pd.read_csv(screening_csv)
    keyed = apply_human_reviews(screening)
    if record_key not in set(keyed["record_key"]):
        raise ValueError("The screening record is no longer available")
    if reviews_csv.exists():
        reviews = pd.read_csv(reviews_csv, dtype=str).fillna("")
        reviews = reviews.reindex(columns=HUMAN_REVIEW_COLUMNS, fill_value="")
    else:
        reviews = pd.DataFrame(columns=HUMAN_REVIEW_COLUMNS)
    reviews = reviews.loc[reviews["record_key"] != record_key].copy()
    if decision:
        reviews.loc[len(reviews)] = {
            "record_key": record_key,
            "human_decision": decision,
            "human_note": note.strip(),
            "human_reviewed_at": datetime.now(timezone.utc).isoformat(),
            "abstract_review_status": "human_reviewed",
            "abstract_auto_reviewed_at": "",
        }
    reviews.to_csv(reviews_csv, index=False)
    reviewed = apply_human_reviews(screening, reviews)
    reviewed.to_csv(reviewed_results_csv, index=False)
    return reviewed


def mark_abstract_ai_accepted(
    screening_csv: Path,
    reviews_csv: Path,
    reviewed_results_csv: Path,
    *,
    record_keys: set[str],
) -> tuple[pd.DataFrame, int]:
    """Mark selected records as reviewed by accepting, not replacing, their AI decisions."""

    screening = pd.read_csv(screening_csv)
    keyed = apply_human_reviews(screening)
    available = set(keyed["record_key"])
    selected = record_keys & available
    if not selected:
        raise ValueError("No matching unreviewed records are available to accept")
    if reviews_csv.exists():
        reviews = pd.read_csv(reviews_csv, dtype=str).fillna("")
        reviews = reviews.reindex(columns=HUMAN_REVIEW_COLUMNS, fill_value="")
    else:
        reviews = pd.DataFrame(columns=HUMAN_REVIEW_COLUMNS)
    human_keys = set(
        reviews.loc[reviews["human_decision"].isin({"include", "exclude", "uncertain"}), "record_key"]
    )
    selected -= human_keys
    if not selected:
        raise ValueError("All matching records already have human decisions")
    reviews = reviews.loc[~reviews["record_key"].isin(selected)].copy()
    reviewed_at = datetime.now(timezone.utc).isoformat()
    additions = pd.DataFrame([
        {
            "record_key": record_key,
            "human_decision": "",
            "human_note": "",
            "human_reviewed_at": "",
            "abstract_review_status": "ai_accepted",
            "abstract_auto_reviewed_at": reviewed_at,
        }
        for record_key in sorted(selected)
    ])
    reviews = pd.concat([reviews, additions], ignore_index=True)
    reviews.to_csv(reviews_csv, index=False)
    reviewed = apply_human_reviews(screening, reviews)
    reviewed.to_csv(reviewed_results_csv, index=False)
    return reviewed, len(selected)


def truncate_screening_record(title: str, abstract: str) -> tuple[str, str, bool, int]:
    """Keep the leading title/abstract characters within the screening input limit."""

    original_length = len(title) + len(abstract)
    truncated_title = title[:MAX_SCREENING_RECORD_LENGTH]
    abstract_allowance = max(0, MAX_SCREENING_RECORD_LENGTH - len(truncated_title))
    truncated_abstract = abstract[:abstract_allowance]
    return truncated_title, truncated_abstract, original_length > MAX_SCREENING_RECORD_LENGTH, original_length


def screen_record(
    title: str,
    abstract: str,
    criteria: str,
    *,
    api_key: str,
    model: str,
    record_identifier: str,
    client: Any | None = None,
) -> ScreeningDecision:
    """Screen one title/abstract through OpenAI Structured Outputs."""

    if not api_key:
        raise ValueError("An OpenAI API key is required")
    title, abstract, was_truncated, _original_length = truncate_screening_record(title, abstract)
    api_client = client or OpenAI(api_key=api_key)
    safety_identifier = "ml-review-" + hashlib.sha256(record_identifier.encode()).hexdigest()[:24]
    response = api_client.responses.parse(
        model=model,
        instructions=(
            "You are assisting a systematic-review team with title and abstract screening. "
            "Apply the supplied criteria exactly. Choose uncertain when the abstract lacks enough information. "
            "For an excluded record, select exactly one broad exclusion category and give a brief, "
            "record-specific exclusion reason. For included or uncertain records, leave both exclusion fields null. "
            "Do not invent study details. AI output is decision support and requires human review.\n\n"
            f"INCLUSION AND EXCLUSION CRITERIA:\n{criteria}"
        ),
        input=(
            f"TITLE:\n{title}\n\n"
            f"ABSTRACT{' (LEADING EXCERPT; INPUT WAS TRUNCATED)' if was_truncated else ''}:\n{abstract}"
        ),
        text_format=ScreeningDecision,
        reasoning={"effort": "low"},
        safety_identifier=safety_identifier,
        store=False,
    )
    if response.output_parsed is None:
        raise ValueError("OpenAI did not return a screening decision")
    return response.output_parsed


def screen_csv(
    input_csv: Path,
    criteria: str,
    output_csv: Path,
    *,
    api_key: str,
    model: str,
    resume: bool = True,
    client: Any | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Screen records with OpenAI and save progress after every response."""

    df = pd.read_csv(input_csv).reset_index(drop=True)
    if df.empty:
        raise ValueError("The selected screening source is empty")
    config_hash = _configuration_hash(criteria, model)
    decision_columns = [f"ai_{name}" for name in ScreeningDecision.model_fields]
    result = df.copy()
    metadata_columns = [
        "ai_model",
        "ai_config_hash",
        "ai_screened_at",
        "ai_input_truncated",
        "ai_input_characters",
        "ai_input_original_characters",
    ]
    for column in decision_columns + metadata_columns:
        result[column] = None

    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        if len(existing) == len(df) and "ai_config_hash" in existing:
            matching = existing["ai_config_hash"].fillna("").eq(config_hash)
            for column in decision_columns + metadata_columns:
                if column in existing:
                    result.loc[matching, column] = existing.loc[matching, column]

    completed = int(result["ai_decision"].notna().sum())
    if progress_callback:
        progress_callback(completed, len(df), "Resuming saved screening decisions" if completed else "Preparing screening")
    for index, row in df.iterrows():
        if pd.notna(result.loc[index, "ai_decision"]):
            continue
        title = "" if pd.isna(row.get("Title")) else str(row.get("Title", ""))
        abstract = "" if pd.isna(row.get("Abstract")) else str(row.get("Abstract", ""))
        truncated_title, truncated_abstract, was_truncated, original_length = truncate_screening_record(title, abstract)
        decision = screen_record(
            title,
            abstract,
            criteria,
            api_key=api_key,
            model=model,
            record_identifier=_record_identifier(row, index),
            client=client,
        )
        for name, value in decision.model_dump().items():
            result.loc[index, f"ai_{name}"] = value
        result.loc[index, "ai_model"] = model
        result.loc[index, "ai_config_hash"] = config_hash
        result.loc[index, "ai_screened_at"] = datetime.now(timezone.utc).isoformat()
        result.loc[index, "ai_input_truncated"] = was_truncated
        result.loc[index, "ai_input_characters"] = len(truncated_title) + len(truncated_abstract)
        result.loc[index, "ai_input_original_characters"] = original_length
        result.to_csv(output_csv, index=False)
        completed += 1
        if progress_callback:
            progress_callback(completed, len(df), f"Screened {completed} of {len(df)} records")
    if progress_callback:
        progress_callback(len(df), len(df), "Screening CSV saved")
    return result
