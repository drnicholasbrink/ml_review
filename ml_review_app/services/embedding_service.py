"""OpenAI embedding generation with incremental, resumable CSV writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

MAX_EMBEDDING_TEXT_LENGTH = 24_000


def make_text(row: pd.Series, title_column: str = "Title", abstract_column: str = "Abstract") -> str:
    """Combine title and abstract text for embedding."""

    title = "" if pd.isna(row.get(title_column)) else str(row.get(title_column, ""))
    abstract = "" if pd.isna(row.get(abstract_column)) else str(row.get(abstract_column, ""))
    return f"{title}\n\n{abstract}".strip()


def _resume_embeddings(source: pd.DataFrame, output_csv: Path, model: str) -> pd.Series:
    embeddings = pd.Series([None] * len(source), dtype="object")
    if not output_csv.exists():
        return embeddings
    existing = pd.read_csv(output_csv)
    if len(existing) != len(source) or "Embedding" not in existing or "EmbeddingModel" not in existing:
        return embeddings
    if not existing["EmbeddingModel"].fillna("").eq(model).all():
        return embeddings
    source_ids = source.get("RecordID", pd.Series(range(len(source)))).astype(str).reset_index(drop=True)
    existing_ids = existing.get("RecordID", pd.Series(range(len(existing)))).astype(str).reset_index(drop=True)
    if not source_ids.equals(existing_ids):
        return embeddings
    valid = existing["Embedding"].fillna("").str.startswith("[")
    embeddings.loc[valid] = existing.loc[valid, "Embedding"]
    return embeddings


def add_embeddings(
    input_csv: Path,
    output_csv: Path,
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    title_column: str = "Title",
    abstract_column: str = "Abstract",
    client: Any | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Generate real OpenAI embeddings, saving progress after every batch."""

    if not api_key:
        raise ValueError("An OpenAI API key is required")
    if batch_size < 1 or batch_size > 2048:
        raise ValueError("Embedding batch size must be between 1 and 2048")
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("The selected record source is empty")
    original_texts = [make_text(row, title_column, abstract_column) for _, row in df.iterrows()]
    if any(not text for text in original_texts):
        raise ValueError("Every record must contain a title or abstract before embedding")
    texts = [text[:MAX_EMBEDDING_TEXT_LENGTH] for text in original_texts]
    input_truncated = [len(text) > MAX_EMBEDDING_TEXT_LENGTH for text in original_texts]
    embeddings = _resume_embeddings(df, output_csv, model)
    api_client = client or OpenAI(api_key=api_key)

    pending = [index for index, value in embeddings.items() if not value]
    completed = len(df) - len(pending)
    if progress_callback:
        progress_callback(completed, len(df), "Resuming saved embeddings" if completed else "Preparing embedding batches")
    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        response = api_client.embeddings.create(
            input=[texts[index] for index in indices],
            model=model,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(indices):
            raise ValueError("OpenAI returned an unexpected number of embeddings")
        for index, item in zip(indices, ordered, strict=True):
            embeddings.iloc[index] = json.dumps(item.embedding)
        progress = df.copy()
        progress["Embedding"] = embeddings
        progress["EmbeddingModel"] = model
        progress["EmbeddingInputTruncated"] = input_truncated
        progress["EmbeddingInputCharacters"] = [len(text) for text in texts]
        progress["EmbeddingInputOriginalCharacters"] = [len(text) for text in original_texts]
        progress.to_csv(output_csv, index=False)
        completed += len(indices)
        if progress_callback:
            progress_callback(completed, len(df), f"Embedded {completed} of {len(df)} records")

    result = df.copy()
    result["Embedding"] = embeddings
    result["EmbeddingModel"] = model
    result["EmbeddingInputTruncated"] = input_truncated
    result["EmbeddingInputCharacters"] = [len(text) for text in texts]
    result["EmbeddingInputOriginalCharacters"] = [len(text) for text in original_texts]
    result.to_csv(output_csv, index=False)
    if progress_callback:
        progress_callback(len(df), len(df), "Embedding artifact saved")
    return result
