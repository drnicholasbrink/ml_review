"""CSV upload, schema mapping, and canonicalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

CANONICAL_COLUMNS = ["RecordID", "Title", "Abstract", "Authors", "Date", "Journal", "DOI"]


def save_uploaded_csv(upload: FileStorage, project_path: Path) -> Path:
    """Validate and atomically save an uploaded CSV."""

    filename = secure_filename(upload.filename or "uploaded_source.csv") or "uploaded_source.csv"
    if not filename.lower().endswith(".csv"):
        raise ValueError("Upload must be a .csv file")
    destination = project_path / "uploaded_source.csv"
    temporary = project_path / ".uploaded_source.pending.csv"
    upload.save(temporary)
    try:
        preview = pd.read_csv(temporary, nrows=1)
        if not len(preview.columns):
            raise ValueError("Uploaded CSV must contain a header row")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty or is not a readable CSV") from exc
    except ValueError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return destination


def read_csv_preview(csv_path: Path, rows: int = 20) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV preview and return the preview frame and column names."""

    df = pd.read_csv(csv_path, nrows=rows)
    return df, list(df.columns)


def profile_csv(csv_path: Path, unique_id_column: str | None = None) -> dict[str, object]:
    """Return row, column, missing-value, and duplicate-ID profile details."""

    df = pd.read_csv(csv_path)
    profile: dict[str, object] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
    }
    if unique_id_column and unique_id_column in df.columns:
        profile["duplicate_unique_ids"] = int(df[unique_id_column].astype(str).duplicated().sum())
        profile["missing_unique_ids"] = int(df[unique_id_column].isna().sum())
    return profile


def build_column_mapping(form: dict[str, str], columns: Iterable[str]) -> dict[str, str]:
    """Build and validate a canonical column mapping from submitted form values."""

    available = set(columns)
    mapping: dict[str, str] = {}
    for canonical in CANONICAL_COLUMNS:
        value = form.get(canonical, "").strip()
        if value:
            if value not in available:
                raise ValueError(f"Mapped column {value!r} is not in uploaded CSV")
            mapping[canonical] = value
    if "RecordID" not in mapping:
        raise ValueError("A unique ID column is required")
    if "Title" not in mapping and "Abstract" not in mapping:
        raise ValueError("At least one of Title or Abstract must be mapped")
    return mapping


def normalize_records(csv_path: Path, mapping: dict[str, str], output_path: Path) -> pd.DataFrame:
    """Create a canonical record CSV from an uploaded CSV and mapping."""

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Uploaded CSV must contain at least one record")
    record_ids = df[mapping["RecordID"]]
    if record_ids.isna().any() or record_ids.astype(str).str.strip().eq("").any():
        raise ValueError("The mapped unique ID column contains missing values")
    normalized = pd.DataFrame()
    for canonical, source in mapping.items():
        normalized[canonical] = df[source]
    for column in CANONICAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    extra_columns = [col for col in df.columns if col not in set(mapping.values())]
    for col in extra_columns:
        normalized[f"Source_{col}"] = df[col]
    normalized.to_csv(output_path, index=False)
    return normalized


def save_column_mapping(project_path: Path, mapping: dict[str, str]) -> Path:
    """Persist column mapping as JSON."""

    output = project_path / "column_mapping.json"
    output.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return output


def load_column_mapping(project_path: Path) -> dict[str, str]:
    """Load a stored column mapping if present."""

    path = project_path / "column_mapping.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
