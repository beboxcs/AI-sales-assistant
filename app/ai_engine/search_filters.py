import re

from pydantic import BaseModel, Field

from app.ai_engine.inventory_loader import normalize_money


class SearchFilters(BaseModel):
    location: str | None = None
    property_type: str | None = None
    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    min_area_sqm: float | None = Field(default=None, gt=0)
    max_area_sqm: float | None = Field(default=None, gt=0)
    payment_method: str | None = None
    max_down_payment: float | None = Field(default=None, ge=0)
    min_installment_years: float | None = Field(default=None, gt=0)


LOCATION_ALIASES: dict[str, str] = {
    "new cairo": "New Cairo",
    "tagamoa": "New Cairo",
    "tagamo3": "New Cairo",
    "fifth settlement": "Fifth Settlement",
    "sheikh zayed": "Sheikh Zayed",
    "zayed": "Sheikh Zayed",
    "6 october": "6th of October",
    "6th october": "6th of October",
    "october": "6th of October",
    "new capital": "New Capital",
    "capital": "New Capital",
    "north coast": "North Coast",
    "sahel": "North Coast",
}

PAYMENT_METHOD_ALIASES: dict[str, str] = {
    "cash": "Cash",
    "installment": "Installment",
    "installments": "Installment",
    "installment plan": "Installment",
}

_MONEY_PATTERN = (
    r"(?:egp\s*)?\d+(?:\.\d+)?(?:,\d{3})*"
    r"(?:\s*(?:m|million))?"
)


def parse_search_filters(text: str) -> SearchFilters:
    normalized_text = _normalize_text(text)

    bedrooms = _extract_integer_count(
        normalized_text,
        labels=["br", "bedroom", "bedrooms"],
        field_name="bedroom",
    )
    bathrooms = _extract_integer_count(
        normalized_text,
        labels=["bath", "baths", "bathroom", "bathrooms"],
        field_name="bathroom",
    )
    payment_method = _extract_payment_method(normalized_text)

    return SearchFilters(
        location=_extract_alias(normalized_text, LOCATION_ALIASES),
        property_type=_extract_property_type(normalized_text),
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        min_price=_extract_money_limit(normalized_text, limit_type="min_price"),
        max_price=_extract_money_limit(normalized_text, limit_type="max_price"),
        payment_method=payment_method,
        max_down_payment=_extract_max_down_payment(normalized_text),
        min_installment_years=_extract_min_installment_years(normalized_text),
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_property_type(text: str) -> str | None:
    property_aliases = {
        "twin house": "Twin House",
        "twinhouse": "Twin House",
        "town house": "Townhouse",
        "townhouse": "Townhouse",
        "apartment": "Apartment",
        "apt": "Apartment",
        "villa": "Villa",
        "duplex": "Duplex",
        "penthouse": "Penthouse",
        "chalet": "Chalet",
    }
    return _extract_alias(text, property_aliases)


def _extract_alias(text: str, aliases: dict[str, str]) -> str | None:
    matches = {
        canonical
        for alias, canonical in aliases.items()
        if _contains_phrase(text, alias)
    }

    if len(matches) > 1:
        raise ValueError(f"Conflicting values found: {sorted(matches)}")
    return next(iter(matches), None)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _extract_integer_count(
    text: str,
    labels: list[str],
    field_name: str,
) -> int | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    conflict_pattern = re.compile(rf"\b\d+\s*(?:-|\+)\s*\d*\s*(?:{label_pattern})\b")
    if conflict_pattern.search(text):
        return None

    count_pattern = re.compile(rf"\b(\d+)\s*(?:{label_pattern})\b")
    values = {int(match.group(1)) for match in count_pattern.finditer(text)}

    if len(values) > 1:
        raise ValueError(f"Conflicting {field_name} values found: {sorted(values)}")
    return next(iter(values), None)


def _extract_money_limit(text: str, limit_type: str) -> float | None:
    if limit_type == "max_price":
        patterns = [
            rf"\bmax\s+budget\s+(?:is\s+)?(?P<money>{_MONEY_PATTERN})\b",
            rf"\bbudget\s+(?:is\s+)?(?:up\s+to\s+)?(?P<money>{_MONEY_PATTERN})\b",
            rf"\b(?:under|below)\s+(?P<money>{_MONEY_PATTERN})\b",
        ]
    else:
        patterns = [
            rf"\b(?:above|minimum|at\s+least)\s+(?P<money>{_MONEY_PATTERN})\b",
        ]

    return _extract_first_money_match(text, patterns)


def _extract_max_down_payment(text: str) -> float | None:
    patterns = [
        rf"\bmax\s+down\s+payment\s+(?P<money>{_MONEY_PATTERN})\b",
        rf"\bdown\s+payment\s+up\s+to\s+(?P<money>{_MONEY_PATTERN})\b",
    ]
    return _extract_first_money_match(text, patterns)


def _extract_first_money_match(text: str, patterns: list[str]) -> float | None:
    values: set[float] = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            parsed_value = normalize_money(match.group("money"))
            if isinstance(parsed_value, float):
                values.add(parsed_value)

    if len(values) > 1:
        raise ValueError(f"Conflicting money values found: {sorted(values)}")
    return next(iter(values), None)


def _extract_payment_method(text: str) -> str | None:
    matches = {
        canonical
        for alias, canonical in PAYMENT_METHOD_ALIASES.items()
        if _contains_phrase(text, alias)
    }

    if len(matches) > 1:
        raise ValueError(f"Conflicting payment methods found: {sorted(matches)}")
    return next(iter(matches), None)


def _extract_min_installment_years(text: str) -> float | None:
    patterns = [
        r"\b(\d+(?:\.\d+)?)\s+years?\s+installments?\b",
        r"\binstallments?\s+over\s+(\d+(?:\.\d+)?)\s+years?\b",
        r"\bminimum\s+(\d+(?:\.\d+)?)\s+years?\b",
    ]
    values: set[float] = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            values.add(float(match.group(1)))

    if len(values) > 1:
        raise ValueError(f"Conflicting installment year values found: {sorted(values)}")
    return next(iter(values), None)
