import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field, field_validator


class SearchQueryParseError(ValueError):
    """Raised when user search text contains invalid structured filter syntax."""


class SearchFilters(BaseModel):
    location: str | None = None
    property_type: str | None = None
    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    min_price: Decimal | None = Field(default=None, ge=0)
    max_price: Decimal | None = Field(default=None, ge=0)
    min_area_sqm: Decimal | None = Field(default=None, gt=0)
    max_area_sqm: Decimal | None = Field(default=None, gt=0)
    payment_method: str | None = None
    max_down_payment: Decimal | None = Field(default=None, ge=0)
    min_installment_years: Decimal | None = Field(default=None, gt=0)

    @field_validator(
        "min_price",
        "max_price",
        "min_area_sqm",
        "max_area_sqm",
        "max_down_payment",
        "min_installment_years",
        mode="before",
    )
    @classmethod
    def normalize_business_decimal(cls, raw_number):
        if (
            raw_number is None
            or isinstance(raw_number, Decimal)
            or isinstance(raw_number, bool)
        ):
            return raw_number
        try:
            return Decimal(str(raw_number))
        except (InvalidOperation, ValueError):
            return raw_number


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

_PROPERTY_TYPE_ALIASES: dict[str, str] = {
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
_FILTER_VALUE_STARTS = "|".join(
    re.escape(token)
    for token in sorted({
        alias.split()[0]
        for aliases in (
            LOCATION_ALIASES,
            _PROPERTY_TYPE_ALIASES,
            PAYMENT_METHOD_ALIASES,
        )
        for alias in aliases
    })
)
_DIRECTIVE_SUFFIX = r"(?:required|mandatory|must-have|preferred|flexible)"
_MONEY_BOUNDARY = (
    rf"(?:years?|and|but|with|in|{_FILTER_VALUE_STARTS}|{_DIRECTIVE_SUFFIX}|"
    rf"(?:is|are)\s+{_DIRECTIVE_SUFFIX})"
)
_MONEY_CANDIDATE = (
    r"(?:egp\s*)?[+-]?\d[\d,.]*"
    rf"(?:(?!\s+{_MONEY_BOUNDARY}\b)"
    r"\s*[^\W\d_]\w*)?"
    r"[.,;:!?]?"
)
_VALID_MONEY = re.compile(
    r"(?:egp\s*)?"
    r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s*(?P<scale>million|m))?"
    r"(?:\s*egp)?",
    re.IGNORECASE,
)
_MONEY_SENTENCE_BOUNDARY = re.compile(
    r"(?P<money>.+\d)[.,;:!?]\s+\w+",
    re.IGNORECASE,
)
_AREA_VALUE = r"(?P<area>\S+?)\s*(?:sqm|m2|m²)(?!\w)"
_AREA_BOUND_PATTERNS = (
    (
        "min_area_sqm",
        re.compile(
            rf"\b(?:at\s+least|above|over|minimum(?:\s+area)?)\s+{_AREA_VALUE}",
            re.IGNORECASE,
        ),
    ),
    (
        "max_area_sqm",
        re.compile(
            rf"\b(?:under|below|max(?:imum)?\s+area|area\s+up\s+to)\s+"
            rf"{_AREA_VALUE}",
            re.IGNORECASE,
        ),
    ),
)
_EXPLICIT_AREA_CUE = re.compile(
    r"\b(?:minimum\s+area|max(?:imum)?\s+area|area\s+up\s+to)\b",
    re.IGNORECASE,
)
_VALID_AREA_NUMBER = re.compile(
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)


def parse_search_filters(text: str) -> SearchFilters:
    normalized_text = _normalize_text(text)
    min_area_sqm, max_area_sqm, price_text = _extract_area_limits(normalized_text)

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
        min_price=_extract_money_limit(price_text, limit_type="min_price"),
        max_price=_extract_money_limit(price_text, limit_type="max_price"),
        min_area_sqm=min_area_sqm,
        max_area_sqm=max_area_sqm,
        payment_method=payment_method,
        max_down_payment=_extract_max_down_payment(normalized_text),
        min_installment_years=_extract_min_installment_years(normalized_text),
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_property_type(text: str) -> str | None:
    return _extract_alias(text, _PROPERTY_TYPE_ALIASES)


def _extract_alias(text: str, aliases: dict[str, str]) -> str | None:
    matches = {
        canonical
        for alias, canonical in aliases.items()
        if _contains_phrase(text, alias)
    }

    if len(matches) > 1:
        raise SearchQueryParseError(
            f"Conflicting values found: {sorted(matches)}"
        )
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
    try:
        values = {int(match.group(1)) for match in count_pattern.finditer(text)}
    except ValueError as exc:
        raise SearchQueryParseError(f"Invalid {field_name} value") from exc

    if len(values) > 1:
        raise SearchQueryParseError(
            f"Conflicting {field_name} values found: {sorted(values)}"
        )
    return next(iter(values), None)


def _extract_area_limits(
    text: str,
) -> tuple[Decimal | None, Decimal | None, str]:
    values: dict[str, set[Decimal]] = {
        "min_area_sqm": set(),
        "max_area_sqm": set(),
    }
    spans: list[tuple[int, int]] = []

    for field_name, pattern in _AREA_BOUND_PATTERNS:
        for match in pattern.finditer(text):
            values[field_name].add(_parse_area(match.group("area")))
            spans.append(match.span())

    for cue in _EXPLICIT_AREA_CUE.finditer(text):
        if not any(start <= cue.start() < end for start, end in spans):
            raise SearchQueryParseError("Invalid or unsupported area value")

    for field_name, field_values in values.items():
        if len(field_values) > 1:
            raise SearchQueryParseError(
                f"Conflicting {field_name} values found: {sorted(field_values)}"
            )

    price_text = list(text)
    for start, end in spans:
        price_text[start:end] = " " * (end - start)
    return (
        next(iter(values["min_area_sqm"]), None),
        next(iter(values["max_area_sqm"]), None),
        "".join(price_text),
    )


def _parse_area(raw_area: str) -> Decimal:
    if _VALID_AREA_NUMBER.fullmatch(raw_area) is None:
        raise SearchQueryParseError("Invalid or unsupported area value")
    area = Decimal(raw_area.replace(",", ""))
    if area <= 0:
        raise SearchQueryParseError("Area must be greater than zero")
    return area


def _extract_money_limit(text: str, limit_type: str) -> Decimal | None:
    if limit_type == "max_price":
        patterns = [
            rf"\bmax\s+budget\s+(?:is\s+)?"
            rf"(?P<money>{_MONEY_CANDIDATE})(?!\w)",
            rf"\bbudget\s+(?:is\s+)?(?:up\s+to\s+)?"
            rf"(?P<money>{_MONEY_CANDIDATE})(?!\w)",
            rf"\b(?:under|below)\s+(?P<money>{_MONEY_CANDIDATE})(?!\w)",
        ]
    else:
        patterns = [
            rf"\b(?:above|minimum|at\s+least)\s+"
            rf"(?P<money>{_MONEY_CANDIDATE})(?!\w)",
        ]

    return _extract_first_money_match(text, patterns)


def _extract_max_down_payment(text: str) -> Decimal | None:
    patterns = [
        rf"\bmax\s+down\s+payment\s+(?P<money>{_MONEY_CANDIDATE})(?!\w)",
        rf"\bdown\s+payment\s+up\s+to\s+(?P<money>{_MONEY_CANDIDATE})(?!\w)",
    ]
    return _extract_first_money_match(text, patterns)


def _extract_first_money_match(text: str, patterns: list[str]) -> Decimal | None:
    values: set[Decimal] = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            values.add(_parse_money(match.group("money")))

    if len(values) > 1:
        raise SearchQueryParseError(
            f"Conflicting money values found: {sorted(values)}"
        )
    return next(iter(values), None)


def _parse_money(raw_money: str) -> Decimal:
    sentence = _MONEY_SENTENCE_BOUNDARY.fullmatch(raw_money)
    if sentence is not None:
        raw_money = sentence.group("money")
    elif raw_money[-1] in ".,;:!?":
        raw_money = raw_money[:-1]
    match = _VALID_MONEY.fullmatch(raw_money)
    if match is None:
        raise SearchQueryParseError("Invalid or unsupported money value")

    amount = Decimal(match.group("amount").replace(",", ""))
    if match.group("scale") is None:
        return amount
    amount_tuple = amount.as_tuple()
    return Decimal((
        amount_tuple.sign,
        amount_tuple.digits,
        amount_tuple.exponent + 6,
    ))


def _extract_payment_method(text: str) -> str | None:
    matches = {
        canonical
        for alias, canonical in PAYMENT_METHOD_ALIASES.items()
        if _contains_phrase(text, alias)
    }

    if len(matches) > 1:
        raise SearchQueryParseError(
            f"Conflicting payment methods found: {sorted(matches)}"
        )
    return next(iter(matches), None)


def _extract_min_installment_years(text: str) -> Decimal | None:
    patterns = [
        r"\b(\d+(?:\.\d+)?)\s+years?\s+installments?\b",
        r"\binstallments?\s+over\s+(\d+(?:\.\d+)?)\s+years?\b",
        r"\bminimum\s+(\d+(?:\.\d+)?)\s+years?\b",
    ]
    values: set[Decimal] = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            values.add(Decimal(match.group(1)))

    if len(values) > 1:
        raise SearchQueryParseError(
            f"Conflicting installment year values found: {sorted(values)}"
        )
    return next(iter(values), None)
