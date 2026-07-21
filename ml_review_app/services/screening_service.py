"""AI screening schema and deterministic local screening helper."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude", "uncertain"]
    confidence: Literal["high", "medium", "low"] = "medium"
    exclusion_reason: str | None = None
    reasoning: str = Field(default="Deterministic local screening placeholder")
    population_match: bool = False
    exposure_match: bool = False
    outcome_match: bool = False
    study_design_appropriate: bool = False


def local_screen(title: str, abstract: str, criteria: str) -> ScreeningDecision:
    """Offline screening heuristic for tests and demos; real app can swap OpenAI calls in."""

    haystack = f"{title} {abstract}".lower()
    criteria_terms = [term for term in criteria.lower().replace('\n', ' ').split() if len(term) > 4]
    hits = sum(1 for term in set(criteria_terms) if term in haystack)
    if hits >= 2:
        return ScreeningDecision(decision="include", confidence="medium", reasoning=f"Matched {hits} criteria terms")
    if hits == 1:
        return ScreeningDecision(decision="uncertain", confidence="low", reasoning="Matched one criteria term")
    return ScreeningDecision(decision="exclude", confidence="medium", exclusion_reason="criteria_mismatch", reasoning="No criteria terms matched")


def screen_csv_offline(input_csv: Path, criteria: str, output_csv: Path) -> pd.DataFrame:
    """Screen a CSV with the local deterministic heuristic."""

    df = pd.read_csv(input_csv)
    decisions = []
    for _, row in df.iterrows():
        decision = local_screen(str(row.get("Title", "")), str(row.get("Abstract", "")), criteria)
        decisions.append(decision.model_dump())
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(decisions).add_prefix("ai_")], axis=1)
    out.to_csv(output_csv, index=False)
    return out
