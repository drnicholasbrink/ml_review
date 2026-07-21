"""Embedding helpers with a deterministic offline fallback for tests/local demos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def deterministic_embedding(text: str, dimensions: int = 16) -> list[float]:
    """Create a deterministic pseudo-embedding for tests and no-key demos."""

    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    values = [digest[i % len(digest)] / 255 for i in range(dimensions)]
    return values


def make_text(row: pd.Series, title_column: str = "Title", abstract_column: str = "Abstract") -> str:
    """Combine title and abstract text for embedding."""

    return f"{row.get(title_column, '')}\n\n{row.get(abstract_column, '')}".strip()


def add_embeddings_offline(input_csv: Path, output_csv: Path, title_column: str = "Title", abstract_column: str = "Abstract") -> pd.DataFrame:
    """Add deterministic embeddings to records without external API calls."""

    df = pd.read_csv(input_csv)
    df["Embedding"] = [json.dumps(deterministic_embedding(make_text(row, title_column, abstract_column))) for _, row in df.iterrows()]
    df.to_csv(output_csv, index=False)
    return df
