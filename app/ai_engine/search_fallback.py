import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from app.ai_engine.multilingual_search_interpreter import (
    interpret_multilingual_search_query,
)
from app.ai_engine.search_filters import (
    LOCATION_ALIASES,
    PAYMENT_METHOD_ALIASES,
    PROPERTY_TYPE_ALIASES,
    SearchFilters,
)
from app.ai_engine.search_intent import (
    ParsedSearchIntent,
    SearchAmbiguityKind,
    SearchFieldDirective,
)
from app.ai_engine.search_strategy import ConstraintStrength, SearchField


class FallbackDisposition(str, Enum):
    SKIP = "skip"
    SHADOW_ONLY = "shadow_only"
    PROPOSAL_ELIGIBLE = "proposal_eligible"
    FORBIDDEN = "forbidden"


class FallbackReason(str, Enum):
    DETERMINISTIC_SUPPORTED = "deterministic_supported"
    SUPPORTED_WITH_UNSUPPORTED = "supported_with_unsupported"
    UNSUPPORTED_ONLY = "unsupported_only"
    UNKNOWN_ONLY = "unknown_only"
    VAGUE_NUMERIC = "vague_numeric"
    PARSER_ERROR = "parser_error"
    CONTRADICTORY_BOUNDS = "contradictory_bounds"
    EXCLUSION_NOT_REPRESENTABLE = "exclusion_not_representable"
    MULTIPLE_EXPLICIT_VALUES = "multiple_explicit_values"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    UNREPRESENTABLE_ALTERNATIVE = "unrepresentable_alternative"
    UNRECOGNIZED_SUBJECT = "unrecognized_subject"
    UNCLASSIFIED_AMBIGUITY = "unclassified_ambiguity"
    BLANK_QUERY = "blank_query"
    AUTHORITY_OR_PROMPT_ONLY = "authority_or_prompt_only"


@dataclass(frozen=True)
class FallbackDecision:
    disposition: FallbackDisposition
    reason: FallbackReason


_AMBIGUITY_REASONS = {
    SearchAmbiguityKind.PARSER_ERROR: FallbackReason.PARSER_ERROR,
    SearchAmbiguityKind.CONTRADICTORY_BOUNDS: FallbackReason.CONTRADICTORY_BOUNDS,
    SearchAmbiguityKind.EXCLUSION_NOT_REPRESENTABLE:
        FallbackReason.EXCLUSION_NOT_REPRESENTABLE,
    SearchAmbiguityKind.MULTIPLE_EXPLICIT_VALUES:
        FallbackReason.MULTIPLE_EXPLICIT_VALUES,
    SearchAmbiguityKind.SEMANTIC_CONTRADICTION:
        FallbackReason.SEMANTIC_CONTRADICTION,
    SearchAmbiguityKind.UNREPRESENTABLE_ALTERNATIVE:
        FallbackReason.UNREPRESENTABLE_ALTERNATIVE,
    SearchAmbiguityKind.UNRECOGNIZED_SUBJECT:
        FallbackReason.UNRECOGNIZED_SUBJECT,
    SearchAmbiguityKind.UNCLASSIFIED: FallbackReason.UNCLASSIFIED_AMBIGUITY,
}
_AUTHORITY_OR_PROMPT = re.compile(
    r"\b(?:company(?:_id)?|tenant|scope|authentication|session|price\s*basis)\b|"
    r"\b(?:ignore\s+previous|system\s+prompt|developer\s+message|"
    r"execute\s+sql|call\s+(?:a\s+)?tool)\b",
    re.IGNORECASE,
)


def decide_search_fallback(
    intent: ParsedSearchIntent,
    original_text: str,
) -> FallbackDecision:
    kinds = {ambiguity.kind for ambiguity in intent.ambiguities}
    for kind, reason in _AMBIGUITY_REASONS.items():
        if kind in kinds:
            return FallbackDecision(FallbackDisposition.FORBIDDEN, reason)

    has_supported_filters = any(
        getattr(intent.filters, field.value) is not None for field in SearchField
    )
    if _AUTHORITY_OR_PROMPT.search(original_text):
        if (
            has_supported_filters
            and SearchAmbiguityKind.VAGUE_NUMERIC not in kinds
        ):
            return FallbackDecision(
                FallbackDisposition.SKIP,
                FallbackReason.DETERMINISTIC_SUPPORTED,
            )
        return FallbackDecision(
            FallbackDisposition.FORBIDDEN,
            FallbackReason.AUTHORITY_OR_PROMPT_ONLY,
        )
    if SearchAmbiguityKind.VAGUE_NUMERIC in kinds:
        return FallbackDecision(
            FallbackDisposition.SHADOW_ONLY,
            FallbackReason.VAGUE_NUMERIC,
        )
    if has_supported_filters:
        if intent.unsupported_requirements:
            return FallbackDecision(
                FallbackDisposition.SHADOW_ONLY,
                FallbackReason.SUPPORTED_WITH_UNSUPPORTED,
            )
        return FallbackDecision(
            FallbackDisposition.SKIP,
            FallbackReason.DETERMINISTIC_SUPPORTED,
        )
    if not original_text.strip():
        return FallbackDecision(
            FallbackDisposition.FORBIDDEN,
            FallbackReason.BLANK_QUERY,
        )
    if intent.unsupported_requirements:
        return FallbackDecision(
            FallbackDisposition.SHADOW_ONLY,
            FallbackReason.UNSUPPORTED_ONLY,
        )
    return FallbackDecision(
        FallbackDisposition.PROPOSAL_ELIGIBLE,
        FallbackReason.UNKNOWN_ONLY,
    )


class _ProposalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceSpan(_ProposalModel):
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "EvidenceSpan":
        if self.start >= self.end:
            raise ValueError("evidence start must be before end")
        return self


class ProposedTextValue(_ProposalModel):
    value: StrictStr
    evidence: EvidenceSpan


class ProposedCountValue(_ProposalModel):
    value: StrictInt = Field(ge=0)
    evidence: EvidenceSpan


class ProposedDecimalValue(_ProposalModel):
    value: Decimal
    evidence: EvidenceSpan

    @field_validator("value", mode="before")
    @classmethod
    def reject_binary_float(cls, value):
        if isinstance(value, (float, bool)):
            raise ValueError("decimal proposal values must not use binary floats")
        return value


class ProposedSearchFilters(_ProposalModel):
    location: ProposedTextValue | None = None
    property_type: ProposedTextValue | None = None
    bedrooms: ProposedCountValue | None = None
    bathrooms: ProposedCountValue | None = None
    min_price: ProposedDecimalValue | None = None
    max_price: ProposedDecimalValue | None = None
    min_area_sqm: ProposedDecimalValue | None = None
    max_area_sqm: ProposedDecimalValue | None = None
    payment_method: ProposedTextValue | None = None
    max_down_payment: ProposedDecimalValue | None = None
    min_installment_years: ProposedDecimalValue | None = None


class ProposedSearchDirective(_ProposalModel):
    field: SearchField
    strength: ConstraintStrength | None = None
    preferred: StrictBool = False
    allow_alternative: StrictBool = False
    evidence: EvidenceSpan

    @model_validator(mode="after")
    def validate_semantics(self) -> "ProposedSearchDirective":
        SearchFieldDirective(
            field=self.field,
            strength=self.strength,
            preferred=self.preferred,
            allow_alternative=self.allow_alternative,
        )
        return self


class UntrustedSearchProposal(_ProposalModel):
    filters: ProposedSearchFilters
    directives: tuple[ProposedSearchDirective, ...] = ()


class SearchProposalValidationErrorKind(str, Enum):
    EVIDENCE_OUT_OF_RANGE = "evidence_out_of_range"
    EVIDENCE_BLANK = "evidence_blank"
    EVIDENCE_AMBIGUOUS = "evidence_ambiguous"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    UNSUPPORTED_CATEGORICAL = "unsupported_categorical"
    INVALID_FILTER_VALUE = "invalid_filter_value"
    CONTRADICTORY_BOUNDS = "contradictory_bounds"
    INVALID_DIRECTIVE = "invalid_directive"


class SearchProposalValidationError(ValueError):
    """Raised when an untrusted semantic proposal lacks deterministic support."""

    def __init__(
        self,
        message: str,
        *,
        kind: SearchProposalValidationErrorKind,
        fields: tuple[SearchField, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.fields = fields


_CATEGORICAL_VALUES = {
    SearchField.LOCATION: frozenset(LOCATION_ALIASES.values()),
    SearchField.PROPERTY_TYPE: frozenset(PROPERTY_TYPE_ALIASES.values()),
    SearchField.PAYMENT_METHOD: frozenset(PAYMENT_METHOD_ALIASES.values()),
}


def validate_search_proposal(
    proposal: UntrustedSearchProposal,
    original_text: str,
) -> ParsedSearchIntent:
    filters = _validate_proposed_filters(proposal.filters, original_text)
    directives = _validate_proposed_directives(
        proposal.directives,
        original_text,
        filters,
    )
    try:
        return ParsedSearchIntent(filters=filters, directives=directives)
    except ValidationError as exc:
        raise SearchProposalValidationError(
            "invalid proposed intent",
            kind=SearchProposalValidationErrorKind.INVALID_DIRECTIVE,
            fields=tuple(directive.field for directive in directives),
        ) from exc


def _validate_proposed_filters(
    proposal: ProposedSearchFilters,
    original_text: str,
) -> SearchFilters:
    proposed_values = {}
    for field in SearchField:
        proposed = getattr(proposal, field.value)
        if proposed is None:
            continue
        if (
            field in _CATEGORICAL_VALUES
            and proposed.value not in _CATEGORICAL_VALUES[field]
        ):
            raise SearchProposalValidationError(
                f"unsupported categorical value for {field.value}",
                kind=SearchProposalValidationErrorKind.UNSUPPORTED_CATEGORICAL,
                fields=(field,),
            )
        evidence_intent = _interpret_evidence(
            proposed.evidence,
            original_text,
            field,
        )
        if getattr(evidence_intent.filters, field.value) != proposed.value:
            raise SearchProposalValidationError(
                f"evidence does not support proposed {field.value}",
                kind=SearchProposalValidationErrorKind.EVIDENCE_MISMATCH,
                fields=(field,),
            )
        proposed_values[field.value] = proposed.value

    try:
        filters = SearchFilters(**proposed_values)
    except ValidationError as exc:
        invalid_fields = tuple(
            SearchField(error["loc"][0])
            for error in exc.errors()
            if error["loc"] and error["loc"][0] in SearchField._value2member_map_
        )
        raise SearchProposalValidationError(
            "invalid proposed filter value",
            kind=SearchProposalValidationErrorKind.INVALID_FILTER_VALUE,
            fields=invalid_fields,
        ) from exc
    _validate_proposed_ranges(filters)
    return filters


def _validate_proposed_ranges(filters: SearchFilters) -> None:
    if (
        filters.min_price is not None
        and filters.max_price is not None
        and filters.min_price > filters.max_price
    ):
        raise SearchProposalValidationError(
            "contradictory proposed price bounds",
            kind=SearchProposalValidationErrorKind.CONTRADICTORY_BOUNDS,
            fields=(SearchField.MIN_PRICE, SearchField.MAX_PRICE),
        )
    if (
        filters.min_area_sqm is not None
        and filters.max_area_sqm is not None
        and filters.min_area_sqm > filters.max_area_sqm
    ):
        raise SearchProposalValidationError(
            "contradictory proposed area bounds",
            kind=SearchProposalValidationErrorKind.CONTRADICTORY_BOUNDS,
            fields=(SearchField.MIN_AREA_SQM, SearchField.MAX_AREA_SQM),
        )


def _validate_proposed_directives(
    proposal: tuple[ProposedSearchDirective, ...],
    original_text: str,
    filters: SearchFilters,
) -> list[SearchFieldDirective]:
    directives = []
    for proposed in proposal:
        directive = SearchFieldDirective(
            field=proposed.field,
            strength=proposed.strength,
            preferred=proposed.preferred,
            allow_alternative=proposed.allow_alternative,
        )
        evidence_intent = _interpret_evidence(
            proposed.evidence,
            original_text,
            proposed.field,
        )
        if (
            directive not in evidence_intent.directives
            or getattr(evidence_intent.filters, proposed.field.value)
            != getattr(filters, proposed.field.value)
        ):
            raise SearchProposalValidationError(
                f"evidence does not support proposed {proposed.field.value} directive",
                kind=SearchProposalValidationErrorKind.EVIDENCE_MISMATCH,
                fields=(proposed.field,),
            )
        directives.append(directive)
    return directives


def _interpret_evidence(
    evidence: EvidenceSpan,
    original_text: str,
    field: SearchField,
) -> ParsedSearchIntent:
    if evidence.end > len(original_text):
        raise SearchProposalValidationError(
            "evidence is outside the original query",
            kind=SearchProposalValidationErrorKind.EVIDENCE_OUT_OF_RANGE,
            fields=(field,),
        )
    evidence_text = original_text[evidence.start:evidence.end]
    if not evidence_text.strip():
        raise SearchProposalValidationError(
            "evidence must not be blank",
            kind=SearchProposalValidationErrorKind.EVIDENCE_BLANK,
            fields=(field,),
        )
    intent = interpret_multilingual_search_query(evidence_text)
    if intent.ambiguities:
        raise SearchProposalValidationError(
            "evidence is ambiguous",
            kind=SearchProposalValidationErrorKind.EVIDENCE_AMBIGUOUS,
            fields=(field,),
        )
    return intent
