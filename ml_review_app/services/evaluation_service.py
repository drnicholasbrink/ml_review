"""Screening evaluation summaries and optional human-reference comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from .screening_service import EXCLUSION_CATEGORY_LABELS

DECISIONS = ["include", "uncertain", "exclude"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]
CRITERIA = [
    ("Population", "ai_population_match"),
    ("Exposure", "ai_exposure_match"),
    ("Outcome", "ai_outcome_match"),
    ("Study design", "ai_study_design_appropriate"),
]

LEGACY_EXCLUSION_PATTERNS = {
    "duplicate": ("duplicate", "already included", "overlap"),
    "insufficient_information": (
        "insufficient", "not enough information", "no abstract", "missing abstract", "unclear", "cannot determine",
    ),
    "publication_type": (
        "publication type", "review", "meta-analysis", "meta analysis", "editorial", "comment", "letter",
        "conference abstract", "protocol", "case report", "not original research",
    ),
    "study_design": (
        "study design", "wrong design", "inappropriate design", "cross-sectional", "ecological", "qualitative",
        "methods paper",
    ),
    "population": (
        "population", "animal", "in vitro", "paediatric", "pediatric", "children", "adolescent", "men only",
        "non-pregnant", "not pregnant",
    ),
    "exposure": ("exposure", "intervention", "temperature", "heat", "cold", "air pollution"),
    "outcome": ("outcome", "endpoint"),
}


def _normalized_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _boolean_series(series: pd.Series) -> pd.Series:
    return _normalized_series(series).isin({"true", "1", "yes", "y"})


def normalize_title(value: object) -> str:
    """Normalize a title for exact and fuzzy reference matching."""

    if pd.isna(value):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(value).lower()).split())


def normalize_exclusion_category(category: object, reason: object) -> str:
    """Return a controlled broad category, including for legacy reason-only results."""

    normalized_category = "" if pd.isna(category) else str(category).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_category in EXCLUSION_CATEGORY_LABELS:
        return normalized_category
    category_text = "" if pd.isna(category) else str(category).strip().lower().replace("_", " ")
    reason_text = "" if pd.isna(reason) else str(reason).strip().lower().replace("_", " ")
    legacy_text = f"{category_text} {reason_text}".strip()
    for broad_category, patterns in LEGACY_EXCLUSION_PATTERNS.items():
        if any(pattern in legacy_text for pattern in patterns):
            return broad_category
    return "other"


def build_screening_evaluation(df: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-safe data for every screening decision instrument chart."""

    if df.empty or "ai_decision" not in df.columns:
        raise ValueError("Screening results are empty or missing ai_decision")
    decisions = _normalized_series(df["ai_decision"])
    confidence = _normalized_series(df.get("ai_confidence", pd.Series(index=df.index, dtype="object")))
    decision_counts = {label: int((decisions == label).sum()) for label in DECISIONS}
    confidence_counts = {label: int((confidence == label).sum()) for label in CONFIDENCE_LEVELS}
    total = len(df)

    criteria = []
    for label, column in CRITERIA:
        matches = int(_boolean_series(df.get(column, pd.Series(False, index=df.index))).sum())
        criteria.append({"label": label, "matched": matches, "not_matched": total - matches, "rate": matches / total})

    heatmap = []
    for decision in DECISIONS:
        heatmap.append([int(((decisions == decision) & (confidence == level)).sum()) for level in CONFIDENCE_LEVELS])

    excluded = df[decisions == "exclude"]
    categories = excluded.get("ai_exclusion_category", pd.Series(index=excluded.index, dtype="object"))
    reasons = excluded.get("ai_exclusion_reason", pd.Series(index=excluded.index, dtype="object"))
    normalized_categories = pd.Series(
        [normalize_exclusion_category(category, reasons.get(index)) for index, category in categories.items()],
        index=excluded.index,
        dtype="object",
    )
    exclusion_counts = normalized_categories.value_counts()
    exclusion_categories = [
        {"category": category, "label": EXCLUSION_CATEGORY_LABELS[category], "count": int(count)}
        for category, count in exclusion_counts.items()
    ]

    sankey_labels = ["Screened", "Included", "Uncertain", "Excluded"]
    sankey_sources: list[int] = []
    sankey_targets: list[int] = []
    sankey_values: list[int] = []
    for target, decision in enumerate(DECISIONS, start=1):
        count = decision_counts[decision]
        if count:
            sankey_sources.append(0)
            sankey_targets.append(target)
            sankey_values.append(count)
    for item in exclusion_categories:
        sankey_labels.append(item["label"])
        sankey_sources.append(3)
        sankey_targets.append(len(sankey_labels) - 1)
        sankey_values.append(item["count"])

    tsne_points = []
    if {"TSNE_1", "TSNE_2"}.issubset(df.columns):
        for index, row in df.iterrows():
            if pd.isna(row["TSNE_1"]) or pd.isna(row["TSNE_2"]):
                continue
            tsne_points.append(
                {
                    "x": float(row["TSNE_1"]),
                    "y": float(row["TSNE_2"]),
                    "title": "" if pd.isna(row.get("Title")) else str(row.get("Title", "")),
                    "decision": decisions.loc[index],
                    "confidence": confidence.loc[index],
                    "record_id": "" if pd.isna(row.get("RecordID")) else str(row.get("RecordID", "")),
                }
            )

    return {
        "total": total,
        "decision_counts": decision_counts,
        "confidence_counts": confidence_counts,
        "criteria": criteria,
        "heatmap": {"x": CONFIDENCE_LEVELS, "y": DECISIONS, "z": heatmap},
        "exclusion_categories": exclusion_categories,
        "funnel": {
            "stages": ["Records screened", "Include or manual review", "Included"],
            "values": [total, decision_counts["include"] + decision_counts["uncertain"], decision_counts["include"]],
        },
        "sankey": {
            "labels": sankey_labels,
            "sources": sankey_sources,
            "targets": sankey_targets,
            "values": sankey_values,
        },
        "tsne_points": tsne_points,
        "manual_review_count": int(((decisions == "uncertain") | (confidence == "low")).sum()),
    }


def _decision_column(df: pd.DataFrame) -> str | None:
    aliases = {"decision", "screening_decision", "human_decision", "status", "included", "include"}
    return next((column for column in df.columns if column.strip().lower() in aliases), None)


def _positive_decisions(series: pd.Series) -> pd.Series:
    return _normalized_series(series).isin({"include", "included", "yes", "true", "1", "eligible", "retain"})


def compare_with_human(
    ai_df: pd.DataFrame,
    human_df: pd.DataFrame,
    *,
    threshold: int = 85,
    uncertain_is_positive: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Match titles one-to-one and calculate reference comparison metrics."""

    if "Title" not in ai_df.columns or "Title" not in human_df.columns:
        raise ValueError("Both AI and human CSV files must contain a Title column")
    if not 0 <= threshold <= 100:
        raise ValueError("Title match threshold must be between 0 and 100")

    ai = ai_df.copy().reset_index(drop=True)
    human = human_df.copy().reset_index(drop=True)
    ai["_normalized_title"] = ai["Title"].map(normalize_title)
    human["_normalized_title"] = human["Title"].map(normalize_title)
    human_decision_column = _decision_column(human)
    human["_positive"] = True if human_decision_column is None else _positive_decisions(human[human_decision_column])
    positive_labels = {"include", "uncertain"} if uncertain_is_positive else {"include"}
    ai["_positive"] = _normalized_series(ai["ai_decision"]).isin(positive_labels)

    available = {index: title for index, title in human["_normalized_title"].items() if title}
    rows = []
    for ai_index, ai_row in ai.iterrows():
        title = ai_row["_normalized_title"]
        match = process.extractOne(title, available, scorer=fuzz.token_sort_ratio, score_cutoff=threshold) if title and available else None
        if match:
            _matched_title, score, human_index = match
            human_row = human.loc[human_index]
            available.pop(human_index, None)
            human_positive: bool | None = bool(human_row["_positive"])
            human_title = human_row["Title"]
        else:
            score, human_index, human_positive, human_title = None, None, None, None
        rows.append(
            {
                "ai_index": ai_index,
                "ai_title": ai_row["Title"],
                "ai_decision": ai_row.get("ai_decision", ""),
                "ai_positive": bool(ai_row["_positive"]),
                "human_index": human_index,
                "human_title": human_title,
                "human_positive": human_positive,
                "match_score": score,
                "match_status": "matched" if human_index is not None else "AI record unmatched",
            }
        )

    comparison = pd.DataFrame(rows)
    matched = comparison[comparison["human_index"].notna()]
    tp = int(((matched["ai_positive"] == True) & (matched["human_positive"] == True)).sum())  # noqa: E712
    fp_matched = int(((matched["ai_positive"] == True) & (matched["human_positive"] == False)).sum())  # noqa: E712
    fn_matched = int(((matched["ai_positive"] == False) & (matched["human_positive"] == True)).sum())  # noqa: E712
    tn = int(((matched["ai_positive"] == False) & (matched["human_positive"] == False)).sum())  # noqa: E712
    unmatched_ai_positive = int(((comparison["human_index"].isna()) & comparison["ai_positive"]).sum())
    unmatched_human_indices = list(available)
    unmatched_human_positive = int(human.loc[unmatched_human_indices, "_positive"].sum()) if unmatched_human_indices else 0
    fp = fp_matched + unmatched_ai_positive
    fn = fn_matched + unmatched_human_positive
    sensitivity = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision is not None and sensitivity is not None and precision + sensitivity else None
    specificity = tn / (tn + fp_matched) if human_decision_column is not None and tn + fp_matched else None

    for human_index in unmatched_human_indices:
        human_row = human.loc[human_index]
        comparison.loc[len(comparison)] = {
            "ai_index": None,
            "ai_title": None,
            "ai_decision": None,
            "ai_positive": None,
            "human_index": human_index,
            "human_title": human_row["Title"],
            "human_positive": bool(human_row["_positive"]),
            "match_score": None,
            "match_status": "Human record unmatched",
        }

    metrics = {
        "human_file_type": "full decisions" if human_decision_column else "included-only list",
        "human_decision_column": human_decision_column,
        "uncertain_is_positive": uncertain_is_positive,
        "match_threshold": threshold,
        "ai_records": len(ai),
        "human_records": len(human),
        "matched_records": len(matched),
        "match_coverage": len(matched) / max(len(ai), len(human), 1),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn if human_decision_column else None,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1_score": f1,
        "specificity": specificity,
    }
    return metrics, comparison


def save_evaluation_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
