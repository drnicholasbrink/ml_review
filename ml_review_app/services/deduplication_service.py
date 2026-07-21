"""Deduplication helpers extracted from the notebook workflow."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_text(value: object) -> str:
    """Normalize text for duplicate matching."""

    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_records(
    input_csv: Path,
    output_csv: Path,
    report_csv: Path,
    unique_id_column: str = "RecordID",
    match_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate records and write kept records plus a duplicate report."""

    df = pd.read_csv(input_csv)
    if unique_id_column not in df.columns:
        raise ValueError(f"Unique ID column {unique_id_column!r} not found")
    match_columns = match_columns or (["Title"] if "Title" in df.columns else [unique_id_column])
    for col in match_columns:
        if col not in df.columns:
            raise ValueError(f"Match column {col!r} not found")
    match_key = df[match_columns].fillna("").astype(str).agg(" | ".join, axis=1).map(normalize_text)
    working = df.copy()
    working["_dedupe_key"] = match_key
    working["_duplicate_group_size"] = working.groupby("_dedupe_key")["_dedupe_key"].transform("size")
    kept = working.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key", "_duplicate_group_size"])
    duplicates = working[working["_duplicate_group_size"] > 1].copy()
    if not duplicates.empty:
        duplicates["dedupe_status"] = duplicates.duplicated("_dedupe_key", keep="first").map({False: "kept", True: "dropped"})
    else:
        duplicates["dedupe_status"] = []
    kept.to_csv(output_csv, index=False)
    duplicates.to_csv(report_csv, index=False)
    return kept, duplicates
