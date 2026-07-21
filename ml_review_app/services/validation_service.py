"""Shared validation helpers for user-supplied form values."""

from __future__ import annotations

from datetime import datetime


def validate_text(
    value: str | None,
    label: str,
    max_length: int,
    *,
    required: bool = False,
) -> str:
    """Normalize and length-check a text form value."""

    normalized = (value or "").strip()
    if required and not normalized:
        raise ValueError(f"{label} is required")
    if len(normalized) > max_length:
        raise ValueError(f"{label} must be {max_length:,} characters or fewer")
    return normalized


def parse_bounded_int(
    value: str | None,
    label: str,
    *,
    minimum: int,
    maximum: int,
    default: int | None = None,
) -> int:
    """Parse an integer and enforce an inclusive range."""

    raw_value = (value or "").strip()
    if not raw_value and default is not None:
        return default
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return parsed


def validate_date_range(mindate: str | None, maxdate: str | None) -> tuple[str | None, str | None]:
    """Validate an optional pair of PubMed dates in YYYY/MM/DD format."""

    start = (mindate or "").strip()
    end = (maxdate or "").strip()
    if bool(start) != bool(end):
        raise ValueError("Provide both a start date and an end date, or leave both blank")
    if not start:
        return None, None
    try:
        start_date = datetime.strptime(start, "%Y/%m/%d").date()
        end_date = datetime.strptime(end, "%Y/%m/%d").date()
    except ValueError as exc:
        raise ValueError("Dates must use YYYY/MM/DD format") from exc
    if start_date > end_date:
        raise ValueError("Start date must not be after end date")
    return start, end
