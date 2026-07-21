"""Structured OpenAI screening with incremental, resumable outputs."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

MAX_SCREENING_RECORD_LENGTH = 100_000


class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude", "uncertain"]
    confidence: Literal["high", "medium", "low"]
    exclusion_reason: str | None = None
    reasoning: str = Field(description="Concise rationale grounded in the supplied title, abstract, and criteria")
    population_match: bool
    exposure_match: bool
    outcome_match: bool
    study_design_appropriate: bool


def _configuration_hash(criteria: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{criteria}".encode()).hexdigest()


def _record_identifier(row: pd.Series, index: int) -> str:
    value = row.get("RecordID", index)
    return str(index if pd.isna(value) else value)


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
    record_length = len(title) + len(abstract)
    if record_length > MAX_SCREENING_RECORD_LENGTH:
        raise ValueError(
            f"Record {record_identifier} contains {record_length:,} title/abstract characters; "
            f"shorten it to {MAX_SCREENING_RECORD_LENGTH:,} characters before screening"
        )
    api_client = client or OpenAI(api_key=api_key)
    safety_identifier = "ml-review-" + hashlib.sha256(record_identifier.encode()).hexdigest()[:24]
    response = api_client.responses.parse(
        model=model,
        instructions=(
            "You are assisting a systematic-review team with title and abstract screening. "
            "Apply the supplied criteria exactly. Choose uncertain when the abstract lacks enough information. "
            "Do not invent study details. AI output is decision support and requires human review.\n\n"
            f"INCLUSION AND EXCLUSION CRITERIA:\n{criteria}"
        ),
        input=f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}",
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
) -> pd.DataFrame:
    """Screen records with OpenAI and save progress after every response."""

    df = pd.read_csv(input_csv).reset_index(drop=True)
    if df.empty:
        raise ValueError("The selected screening source is empty")
    config_hash = _configuration_hash(criteria, model)
    decision_columns = [f"ai_{name}" for name in ScreeningDecision.model_fields]
    result = df.copy()
    for column in decision_columns + ["ai_model", "ai_config_hash", "ai_screened_at"]:
        result[column] = None

    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        if len(existing) == len(df) and "ai_config_hash" in existing:
            matching = existing["ai_config_hash"].fillna("").eq(config_hash)
            for column in decision_columns + ["ai_model", "ai_config_hash", "ai_screened_at"]:
                if column in existing:
                    result.loc[matching, column] = existing.loc[matching, column]

    for index, row in df.iterrows():
        if pd.notna(result.loc[index, "ai_decision"]):
            continue
        decision = screen_record(
            "" if pd.isna(row.get("Title")) else str(row.get("Title", "")),
            "" if pd.isna(row.get("Abstract")) else str(row.get("Abstract", "")),
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
        result.to_csv(output_csv, index=False)
    return result
