import re
from decimal import Decimal

from app.ai_engine.search_filters import (
    LOCATION_ALIASES,
    SearchFilters,
    parse_search_filters,
)
from app.ai_engine.search_intent import (
    ParsedSearchIntent,
    SearchAmbiguity,
    SearchFieldDirective,
)
from app.ai_engine.search_strategy import ConstraintStrength, SearchField

_LOCATION = re.compile(
    r"\b(?:" + "|".join(
        re.escape(alias) for alias in sorted(LOCATION_ALIASES, key=len, reverse=True)
    ) + r")\b"
)
_TARGET_PRICE = re.compile(
    r"\b(?:around|about|approximately|target(?:ing)?)\s+"
    r"(?:(?:budget|price)\s+)?(?:(?:is|of)\s+)?(?:egp\s*)?"
    r"\d[\d,]*(?:\.\d+)?(?:\s*(?:million|m|egp))?(?![\w.])"
    r"(?!\s*(?:years?|bedrooms?|bathrooms?|br|sqm|m2|sqft|square)\b)"
)
_TARGET_COUNT = re.compile(
    r"\b(?:around|about|approximately|target(?:ing)?)\s+\d+(?:\.\d+)?\s+"
    r"(?:(?P<min_installment_years>years?)|(?P<bedrooms>bedrooms?|br)|"
    r"(?P<bathrooms>bathrooms?)|(?P<min_area_sqm>sqm|m2|sqft|square\s+meters?))\b"
)
_MINIMUM_YEARS = re.compile(r"\bminimum\s+(\d+(?:\.\d+)?)\s+years?\b")
_CLAUSE_BREAK = re.compile(r"\b(?:but|and)\b|[;!?]|\.(?!\d)|,(?!\d)")
_SEMANTICS = (
    ("hard", r"\b(?:must(?:\s+(?:be|have))?|must-have|only|required?|mandatory)\s+(.+)",
     r"(.+?)\s+(?:(?:is|are)\s+)?(?:required|mandatory|must-have)\b"),
    ("preferred", r"\bprefer(?:red)?\s+(.+)",
     r"(.+?)\s+(?:(?:is|are)\s+)?preferred\b"),
    ("alternative", r"\bflexible\s+(?:on|about|with)\s+(.+)",
     r"(.+?)\s+(?:(?:is|are)\s+)?flexible\b"),
)
_FIELD_LABELS = {
    SearchField.LOCATION: r"\b(?:location|area)s?\b",
    SearchField.PROPERTY_TYPE: r"\bproperty\s+types?\b",
    SearchField.BEDROOMS: r"\b(?:bedrooms?|br)\b",
    SearchField.BATHROOMS: r"\bbath(?:room)?s?\b",
    SearchField.MIN_PRICE: r"\b(?:price|budget)\b",
    SearchField.MAX_PRICE: r"\b(?:price|budget)\b",
    SearchField.PAYMENT_METHOD: r"\bpayment\s+methods?\b",
    SearchField.MAX_DOWN_PAYMENT: r"\bdown\s+payment\b",
    SearchField.MIN_INSTALLMENT_YEARS: r"\b(?:years?|installments?)\b",
}
_GENERIC_ALTERNATIVES = {
    SearchField.LOCATION: r"other\s+locations?",
    SearchField.PROPERTY_TYPE: r"other\s+property\s+types?",
    SearchField.PAYMENT_METHOD: r"other\s+payment\s+methods?",
}
_VAGUE_NUMERICS = (
    (SearchField.MIN_AREA_SQM, r"\b(?:large|spacious|big)\s+(?:apartment|unit|villa|home)\b",
     "Area needs a precise numeric value."),
    (SearchField.MAX_PRICE, r"\b(?:cheap|affordable)\s+(?:apartment|unit|villa|home)\b",
     "Price needs a precise numeric value."),
    (SearchField.MAX_DOWN_PAYMENT, r"\blow\s+down\s+payment\b",
     "Down payment needs a precise numeric value."),
    (SearchField.MIN_INSTALLMENT_YEARS, r"\blong\s+installments?\b",
     "Installment duration needs a precise numeric value."),
)
_UNSUPPORTED = (
    ("sea view", r"\bsea[ -]view\b"),
    ("near school", r"\bnear\s+(?:a\s+|the\s+)?schools?\b"),
    ("near metro", r"\bnear\s+(?:a\s+|the\s+)?metro\b"),
    ("quiet area", r"\bquiet\s+area\b"),
    ("luxury finishing", r"\bluxury\s+finishing\b"),
    ("good investment", r"\b(?:good\s+investment|investment\s+(?:property|opportunity))\b"),
)


def interpret_search_query(text: str) -> ParsedSearchIntent:
    """Interpret explicit English intent without choosing a search policy."""
    normalized = " ".join(text.lower().split())
    if not normalized:
        raise ValueError("search query must not be empty")

    ambiguities: list[SearchAmbiguity] = []
    filters = _extract_filters(normalized, ambiguities)
    directives = _detect_directives(normalized, filters, ambiguities)
    for field, pattern, reason in _VAGUE_NUMERICS:
        if getattr(filters, field.value) is None and re.search(pattern, normalized):
            ambiguities.append(SearchAmbiguity(field=field, reason=reason))
    field_order = {field: index for index, field in enumerate(SearchField)}
    return ParsedSearchIntent(
        filters=filters,
        directives=directives,
        ambiguities=sorted(
            ambiguities,
            key=lambda ambiguity: (
                field_order.get(ambiguity.field, len(field_order)), ambiguity.reason
            ),
        ),
        unsupported_requirements=[
            label for label, pattern in _UNSUPPORTED if re.search(pattern, normalized)
        ],
    )


def _try_parse(text: str) -> SearchFilters | None:
    try:
        return parse_search_filters(text)
    except ValueError:
        return None


def _extract_filters(text: str, ambiguities: list[SearchAmbiguity]) -> SearchFilters:
    baseline = _try_parse(text)
    corrected = text
    for match in _TARGET_COUNT.finditer(text):
        ambiguities.append(SearchAmbiguity(
            field=SearchField(match.lastgroup),
            reason="An approximate value does not specify an exact requirement or bound.",
        ))
    corrected = _TARGET_COUNT.sub(" ", corrected)
    if _TARGET_PRICE.search(text):
        ambiguities.append(SearchAmbiguity(
            field=SearchField.MAX_PRICE,
            reason="A target price does not specify a minimum or maximum price.",
        ))
        corrected = _TARGET_PRICE.sub(" ", corrected)
    locations = {LOCATION_ALIASES[match[0]] for match in _LOCATION.finditer(text)}
    if len(locations) > 1:
        ambiguities.append(SearchAmbiguity(
            field=SearchField.LOCATION,
            reason="Multiple explicit location alternatives are not representable by the current contract.",
        ))
        corrected = _LOCATION.sub(" ", corrected)
    filters = _try_parse(corrected) if corrected != text else baseline
    if filters is None:
        ambiguities.append(SearchAmbiguity(
            reason="The query contains conflicting or invalid structured filter values.",
        ))
        return SearchFilters()
    return _suppress_year_price(corrected, filters)


def _suppress_year_price(text: str, filters: SearchFilters) -> SearchFilters:
    false_prices = [
        match for match in _MINIMUM_YEARS.finditer(text)
        if filters.min_price == Decimal(match[1])
        and not re.search(r"\b(?:price|budget|cost)\s*$", text[:match.start()])
    ]
    if not false_prices:
        return filters
    remainder = text
    for match in reversed(false_prices):
        remainder = remainder[:match.start()] + " " + remainder[match.end():]
    remaining_filters = _try_parse(remainder)
    return filters.model_copy(update={
        "min_price": remaining_filters.min_price if remaining_filters else None,
    })


def _referenced_fields(subject: str, filters: SearchFilters) -> list[SearchField]:
    if _TARGET_PRICE.search(subject) or _TARGET_COUNT.search(subject):
        return []
    for _, pattern in _UNSUPPORTED:
        subject = re.sub(pattern, " ", subject)
    mentioned = _try_parse(subject)
    if mentioned is None:
        return []
    mentioned_fields = [
        field for field in SearchField if getattr(mentioned, field.value) is not None
    ]
    if mentioned_fields:
        return [
            field for field in mentioned_fields
            if getattr(mentioned, field.value) == getattr(filters, field.value)
        ]
    return [
        field for field, pattern in _FIELD_LABELS.items()
        if getattr(filters, field.value) is not None and re.search(pattern, subject)
    ]


def _detect_directives(
    text: str, filters: SearchFilters, ambiguities: list[SearchAmbiguity]
) -> list[SearchFieldDirective]:
    flags: dict[SearchField, set[str]] = {}
    for clause in _CLAUSE_BREAK.split(text):
        for semantic, prefix, suffix in _SEMANTICS:
            match = re.search(prefix, clause) or re.search(suffix, clause)
            if match is None:
                continue
            fields = _referenced_fields(match[1], filters)
            if len(fields) != 1:
                ambiguities.append(SearchAmbiguity(
                    reason="Semantic wording does not unambiguously identify a supported requirement.",
                ))
                continue
            flags.setdefault(fields[0], set()).add(
                "ambiguous" if re.search(r"\b(?:not|never|no)\b", clause) else semantic
            )
        for field, subject in _GENERIC_ALTERNATIVES.items():
            if re.search(rf"\b{subject}\s+(?:are\s+)?(?:not\s+)?(?:okay|ok|acceptable)\b", clause):
                if getattr(filters, field.value) is not None:
                    flags.setdefault(field, set()).add(
                        "ambiguous" if re.search(r"\b(?:not|except|only)\b", clause)
                        else "alternative"
                    )
    return _directives_from_flags(flags, ambiguities)


def _directives_from_flags(
    flags: dict[SearchField, set[str]], ambiguities: list[SearchAmbiguity]
) -> list[SearchFieldDirective]:
    directives = []
    for field in SearchField:
        semantics = flags.get(field)
        if not semantics:
            continue
        if "ambiguous" in semantics:
            ambiguities.append(SearchAmbiguity(
                field=field,
                reason="Negated or restricted semantic wording needs clarification.",
            ))
            continue
        if "hard" in semantics and len(semantics) > 1:
            ambiguities.append(SearchAmbiguity(
                field=field,
                reason="A required field cannot also be preferred or flexible.",
            ))
            continue
        directives.append(SearchFieldDirective(
            field=field,
            strength=ConstraintStrength.HARD if "hard" in semantics else None,
            preferred="preferred" in semantics,
            allow_alternative="alternative" in semantics,
        ))
    return directives
