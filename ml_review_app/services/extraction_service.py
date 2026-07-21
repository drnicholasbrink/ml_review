"""Structured extraction schema placeholder for the Flask migration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DataExtraction(BaseModel):
    population_description: str | None = None
    country: str | None = None
    exposures: list[dict] = Field(default_factory=list)
    outcomes: list[dict] = Field(default_factory=list)
    effect_estimates: list[dict] = Field(default_factory=list)
    extraction_confidence: str = "not_run"
