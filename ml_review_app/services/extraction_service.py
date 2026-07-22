"""Structured, resumable OpenAI data extraction for included studies."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from .screening_service import screening_record_key

MAX_EXTRACTION_RECORD_LENGTH = 100_000
EXTRACTION_SCHEMA_VERSION = 2


class GeographicLocation(BaseModel):
    city: str | None = None
    province_state: str | None = None
    country: str | None = None
    region: str | None = None
    multi_location: bool = False


class ExposureMetric(BaseModel):
    metric_type: str
    specific_measure: str | None = None
    unit: str | None = None
    lag_structure: str | None = None


class OutcomeMeasure(BaseModel):
    outcome_type: str
    specific_outcome: str | None = None
    age_group: str | None = None


class EffectEstimate(BaseModel):
    estimate_type: str
    value: str
    confidence_interval: str | None = None
    p_value: str | None = None
    comparison: str | None = None
    subgroup: str | None = None


class DataExtraction(BaseModel):
    population_description: str
    age_range: str | None = None
    sex_distribution: str | None = None
    vulnerable_groups: list[str] = Field(default_factory=list)
    location: GeographicLocation
    study_period_start: str | None = None
    study_period_end: str | None = None
    season: str | None = None
    exposures: list[ExposureMetric] = Field(default_factory=list)
    exposure_source: str | None = None
    outcomes: list[OutcomeMeasure] = Field(default_factory=list)
    outcome_source: str | None = None
    sample_size: str | None = None
    study_design: str
    study_design_details: str | None = None
    statistical_methods: list[str] = Field(default_factory=list)
    confounders_adjusted: list[str] = Field(default_factory=list)
    effect_estimates: list[EffectEstimate] = Field(default_factory=list)
    key_finding_summary: str
    data_completeness: Literal["complete", "partial", "limited"]
    extraction_confidence: Literal["high", "medium", "low"]
    notes: str | None = None


def _record_key(row: pd.Series, index: int) -> str:
    value = row.get("record_key")
    return str(value) if pd.notna(value) and str(value).strip() else screening_record_key(row, index)


def _configuration_hash(criteria: str, model: str, source_hash: str = "abstract") -> str:
    payload = (
        f"schema={EXTRACTION_SCHEMA_VERSION}\nmodel={model}\nlimit={MAX_EXTRACTION_RECORD_LENGTH}"
        f"\nsource={source_hash}\n{criteria}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def extraction_candidates(screening_df: pd.DataFrame, *, include_uncertain: bool) -> pd.DataFrame:
    if screening_df.empty or "ai_decision" not in screening_df.columns:
        raise ValueError("Screening results are empty or missing ai_decision")
    screening_df = screening_df.reset_index(drop=True).copy()
    if "record_key" not in screening_df:
        screening_df["record_key"] = [screening_record_key(row, index) for index, row in screening_df.iterrows()]
    decisions = screening_df.get("final_decision", screening_df["ai_decision"]).fillna("").astype(str).str.lower()
    allowed = {"include", "uncertain"} if include_uncertain else {"include"}
    return screening_df[decisions.isin(allowed)].copy().reset_index(drop=True)


def extract_record(
    title: str,
    abstract: str,
    criteria: str,
    *,
    api_key: str,
    model: str,
    record_identifier: str,
    full_text_pdf: Path | None = None,
    client: Any | None = None,
) -> DataExtraction:
    """Extract explicitly reported data from a PDF when available, otherwise the abstract."""

    if not api_key:
        raise ValueError("An OpenAI API key is required")
    title = title[:MAX_EXTRACTION_RECORD_LENGTH]
    abstract_allowance = max(0, MAX_EXTRACTION_RECORD_LENGTH - len(title))
    abstract_excerpt = abstract[:abstract_allowance]
    truncated = len(title) + len(abstract) > MAX_EXTRACTION_RECORD_LENGTH
    api_client = client or OpenAI(api_key=api_key)
    source_description = "supplied full-text PDF and abstract" if full_text_pdf else "supplied title and abstract"
    text_input = (
        f"TITLE:\n{title}\n\n"
        f"ABSTRACT{' (LEADING EXCERPT; INPUT WAS TRUNCATED)' if truncated else ''}:\n{abstract_excerpt}"
    )
    response_input: Any = text_input
    if full_text_pdf is not None:
        encoded_pdf = base64.b64encode(full_text_pdf.read_bytes()).decode("ascii")
        response_input = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": text_input},
                {
                    "type": "input_file",
                    "filename": full_text_pdf.name,
                    "file_data": f"data:application/pdf;base64,{encoded_pdf}",
                },
            ],
        }]
    response = api_client.responses.parse(
        model=model,
        instructions=(
            "You are assisting a systematic-review team with structured data extraction. "
            f"Extract only information explicitly stated in the {source_description}. "
            "Do not infer missing details. Preserve effect estimates and uncertainty intervals in their reported form. "
            "Use empty lists or null values when information is absent, and lower confidence/completeness when the abstract is insufficient. "
            "The result is decision support and must be validated by a human reviewer.\n\n"
            f"REVIEW CRITERIA:\n{criteria}"
        ),
        input=response_input,
        text_format=DataExtraction,
        reasoning={"effort": "low"},
        safety_identifier="ml-review-extract-" + hashlib.sha256(record_identifier.encode()).hexdigest()[:24],
        store=False,
    )
    if response.output_parsed is None:
        raise ValueError("OpenAI did not return structured extraction data")
    return response.output_parsed


def _flatten_result(
    row: pd.Series,
    extraction: DataExtraction,
    *,
    key: str,
    model: str,
    config_hash: str,
    full_text_pdf: Path | None,
) -> dict[str, Any]:
    value = extraction.model_dump()
    location = value["location"]
    base_columns = [
        "RecordID", "PMID", "DOI", "Title", "Abstract", "Authors", "Date", "Journal",
        "ai_decision", "ai_confidence", "ai_reasoning", "final_decision", "final_decision_source",
    ]
    result = {column: row.get(column, "") for column in base_columns if column in row.index}
    result.update(
        {
            "extraction_record_key": key,
            "population_description": value["population_description"],
            "age_range": value["age_range"],
            "sex_distribution": value["sex_distribution"],
            "vulnerable_groups": json.dumps(value["vulnerable_groups"], ensure_ascii=False),
            "city": location["city"],
            "province_state": location["province_state"],
            "country": location["country"],
            "region": location["region"],
            "multi_location": location["multi_location"],
            "study_period_start": value["study_period_start"],
            "study_period_end": value["study_period_end"],
            "season": value["season"],
            "exposures": json.dumps(value["exposures"], ensure_ascii=False),
            "exposure_source": value["exposure_source"],
            "outcomes": json.dumps(value["outcomes"], ensure_ascii=False),
            "outcome_source": value["outcome_source"],
            "sample_size": value["sample_size"],
            "study_design": value["study_design"],
            "study_design_details": value["study_design_details"],
            "statistical_methods": json.dumps(value["statistical_methods"], ensure_ascii=False),
            "confounders_adjusted": json.dumps(value["confounders_adjusted"], ensure_ascii=False),
            "effect_estimates": json.dumps(value["effect_estimates"], ensure_ascii=False),
            "key_finding_summary": value["key_finding_summary"],
            "data_completeness": value["data_completeness"],
            "extraction_confidence": value["extraction_confidence"],
            "notes": value["notes"],
            "text_source": "full_text_pdf" if full_text_pdf else "abstract",
            "full_text_filename": full_text_pdf.name if full_text_pdf else None,
            "extraction_model": model,
            "extraction_config_hash": config_hash,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "extraction_json": json.dumps(value, ensure_ascii=False),
        }
    )
    return result


def extract_csv(
    screening_csv: Path,
    criteria: str,
    output_csv: Path,
    *,
    api_key: str,
    model: str,
    include_uncertain: bool = False,
    limit: int | None = None,
    resume: bool = True,
    client: Any | None = None,
    full_text_files: dict[str, Path] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Extract eligible records with progress saved after every response."""

    screening_df = pd.read_csv(screening_csv)
    candidates = extraction_candidates(screening_df, include_uncertain=include_uncertain)
    candidate_count = len(candidates)
    if not candidate_count:
        raise ValueError("No included records are available for extraction")
    batch = candidates.head(limit).copy() if limit is not None else candidates
    candidate_keys = {_record_key(row, index) for index, row in candidates.iterrows()}
    full_text_files = full_text_files or {}
    expected_configs: dict[str, str] = {}
    for index, row in candidates.iterrows():
        key = _record_key(row, index)
        full_text_pdf = full_text_files.get(key)
        if full_text_pdf and full_text_pdf.is_file():
            source_bytes = full_text_pdf.read_bytes()
        else:
            title = "" if pd.isna(row.get("Title")) else str(row.get("Title", ""))
            abstract = "" if pd.isna(row.get("Abstract")) else str(row.get("Abstract", ""))
            source_bytes = f"{title}\n{abstract}".encode()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        expected_configs[key] = _configuration_hash(criteria, model, source_hash)
    existing = pd.DataFrame()
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        if {"extraction_config_hash", "extraction_record_key"}.issubset(existing.columns):
            existing_keys = existing["extraction_record_key"].fillna("").astype(str)
            matches = [
                key in candidate_keys and str(config) == expected_configs.get(key)
                for key, config in zip(existing_keys, existing["extraction_config_hash"], strict=False)
            ]
            existing = existing.loc[matches].drop_duplicates("extraction_record_key", keep="last").copy()
        else:
            existing = pd.DataFrame()
    processed = set(existing.get("extraction_record_key", pd.Series(dtype="object")).fillna("").astype(str))
    rows = existing.to_dict(orient="records")
    batch_keys = [_record_key(row, index) for index, row in batch.iterrows()]
    completed = sum(key in processed for key in batch_keys)
    if progress_callback:
        progress_callback(completed, len(batch), "Resuming saved extractions" if completed else "Preparing structured extraction")
    for index, row in batch.iterrows():
        key = _record_key(row, index)
        if key in processed:
            continue
        title = "" if pd.isna(row.get("Title")) else str(row.get("Title", ""))
        abstract = "" if pd.isna(row.get("Abstract")) else str(row.get("Abstract", ""))
        full_text_pdf = full_text_files.get(key)
        if full_text_pdf is not None and not full_text_pdf.is_file():
            full_text_pdf = None
        extraction = extract_record(
            title,
            abstract,
            criteria,
            api_key=api_key,
            model=model,
            record_identifier=key,
            full_text_pdf=full_text_pdf,
            client=client,
        )
        rows.append(_flatten_result(
            row,
            extraction,
            key=key,
            model=model,
            config_hash=expected_configs[key],
            full_text_pdf=full_text_pdf,
        ))
        processed.add(key)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        completed += 1
        if progress_callback:
            progress_callback(completed, len(batch), f"Extracted {completed} of {len(batch)} records")
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No extraction results were produced")
    result.to_csv(output_csv, index=False)
    if progress_callback:
        progress_callback(len(batch), len(batch), "Extraction artifacts ready")
    return result, candidate_count


def write_extraction_exports(result: pd.DataFrame, project_path: Path) -> dict[str, str]:
    """Write nested JSON and publication-oriented derived tables."""

    json_rows = []
    for _, row in result.iterrows():
        extracted = json.loads(row["extraction_json"])
        json_rows.append(
            {
                "RecordID": None if pd.isna(row.get("RecordID")) else row.get("RecordID"),
                "PMID": None if pd.isna(row.get("PMID")) else row.get("PMID"),
                "Title": row.get("Title", ""),
                "extraction": extracted,
                "metadata": {
                    "model": row.get("extraction_model"),
                    "timestamp": row.get("extraction_timestamp"),
                    "text_source": row.get("text_source"),
                },
            }
        )
    full_json = project_path / "ai_extraction_full_results.json"
    full_json.write_text(json.dumps(json_rows, indent=2, ensure_ascii=False, default=str) + "\n")

    characteristic_columns = [
        "PMID", "RecordID", "Authors", "Date", "Title", "country", "city", "study_period_start",
        "study_period_end", "study_design", "sample_size", "population_description", "extraction_confidence",
    ]
    characteristics = project_path / "study_characteristics.csv"
    result[[column for column in characteristic_columns if column in result.columns]].to_csv(characteristics, index=False)

    effect_rows = []
    for _, row in result.iterrows():
        for effect in json.loads(row.get("effect_estimates") or "[]"):
            effect_rows.append(
                {
                    "PMID": row.get("PMID", ""),
                    "RecordID": row.get("RecordID", ""),
                    "Title": row.get("Title", ""),
                    **effect,
                }
            )
    effects = project_path / "effect_estimates.csv"
    pd.DataFrame(effect_rows, columns=["PMID", "RecordID", "Title", "estimate_type", "value", "confidence_interval", "p_value", "comparison", "subgroup"]).to_csv(effects, index=False)

    summary = project_path / "extraction_summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_articles": len(result),
                "countries": int(result.get("country", pd.Series(dtype="object")).nunique()),
                "study_designs": result.get("study_design", pd.Series(dtype="object")).value_counts().to_dict(),
                "extraction_confidence": result.get("extraction_confidence", pd.Series(dtype="object")).value_counts().to_dict(),
                "data_completeness": result.get("data_completeness", pd.Series(dtype="object")).value_counts().to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "ai_extraction_full_results_json": full_json.name,
        "study_characteristics": characteristics.name,
        "effect_estimates": effects.name,
        "extraction_summary": summary.name,
    }
