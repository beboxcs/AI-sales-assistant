from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from pydantic import ValidationError

from app.ai_engine.search_fallback import (
    FallbackDisposition,
    FallbackReason,
    SearchProposalValidationError,
    SearchProposalValidationErrorKind,
    UntrustedSearchProposal,
    decide_search_fallback,
    validate_search_proposal,
)
from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_strategy import SearchField


@dataclass(frozen=True)
class SearchFallbackProviderRequest:
    text: str


@dataclass(frozen=True)
class SearchFallbackProviderResponse:
    proposal_json: str | bytes


class SearchFallbackProvider(Protocol):
    def propose_search(
        self,
        *,
        request: SearchFallbackProviderRequest,
    ) -> SearchFallbackProviderResponse | None: ...


class SearchFallbackProviderError(Exception):
    """Raised when a fallback provider cannot return a proposal."""


class SearchFallbackProviderTimeoutError(SearchFallbackProviderError):
    """Raised when a fallback provider exceeds its deadline."""


class SearchFallbackProviderStatus(str, Enum):
    NOT_CALLED = "not_called"
    NO_PROPOSAL = "no_proposal"
    TIMEOUT = "timeout"
    ERROR = "error"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"
    PROPOSAL_REJECTED = "proposal_rejected"
    VALIDATED = "validated"


class SearchFallbackShadowRejectionReason(str, Enum):
    NO_PROPOSAL = "no_proposal"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"


@dataclass(frozen=True)
class ValidatedSearchProposalSummary:
    populated_filter_fields: tuple[str, ...]
    directive_fields: tuple[str, ...]


@dataclass(frozen=True)
class SearchFallbackShadowResult:
    disposition: FallbackDisposition
    decision_reason: FallbackReason
    provider_called: bool
    provider_status: SearchFallbackProviderStatus
    schema_valid: bool | None = None
    proposal_valid: bool | None = None
    evidence_valid: bool | None = None
    exact_filter_match: bool | None = None
    exact_directive_match: bool | None = None
    exact_semantic_match: bool | None = None
    proposal_only_fields: tuple[str, ...] = ()
    deterministic_only_fields: tuple[str, ...] = ()
    disagreed_fields: tuple[str, ...] = ()
    hallucinated_fields: tuple[str, ...] = ()
    unsupported_invention: bool = False
    deterministic_unsupported_requirements: tuple[str, ...] = ()
    rejection_reason: (
        SearchFallbackShadowRejectionReason
        | SearchProposalValidationErrorKind
        | None
    ) = None
    validated_proposal_summary: ValidatedSearchProposalSummary | None = None


@dataclass(frozen=True)
class _SemanticComparison:
    exact_filter_match: bool
    exact_directive_match: bool
    proposal_only_fields: tuple[str, ...]
    deterministic_only_fields: tuple[str, ...]
    disagreed_fields: tuple[str, ...]


_EVIDENCE_REJECTIONS = frozenset({
    SearchProposalValidationErrorKind.EVIDENCE_OUT_OF_RANGE,
    SearchProposalValidationErrorKind.EVIDENCE_BLANK,
    SearchProposalValidationErrorKind.EVIDENCE_AMBIGUOUS,
    SearchProposalValidationErrorKind.EVIDENCE_MISMATCH,
})
_HALLUCINATION_REJECTIONS = _EVIDENCE_REJECTIONS | {
    SearchProposalValidationErrorKind.UNSUPPORTED_CATEGORICAL,
}


def evaluate_search_fallback_shadow(
    *,
    original_text: str,
    deterministic_intent: ParsedSearchIntent,
    provider: SearchFallbackProvider,
) -> SearchFallbackShadowResult:
    decision = decide_search_fallback(deterministic_intent, original_text)
    shadow_result = _initial_result(
        decision.disposition,
        decision.reason,
        deterministic_intent,
    )
    if decision.disposition in {
        FallbackDisposition.SKIP,
        FallbackDisposition.FORBIDDEN,
    }:
        return shadow_result

    provider_response, provider_status, rejection_reason = _invoke_provider(
        provider,
        original_text,
    )
    if provider_status is not None:
        return replace(
            shadow_result,
            provider_called=True,
            provider_status=provider_status,
            rejection_reason=rejection_reason,
        )
    assert provider_response is not None
    return _evaluate_response(
        shadow_result,
        provider_response,
        original_text,
        deterministic_intent,
    )


def _initial_result(
    disposition: FallbackDisposition,
    reason: FallbackReason,
    intent: ParsedSearchIntent,
) -> SearchFallbackShadowResult:
    return SearchFallbackShadowResult(
        disposition=disposition,
        decision_reason=reason,
        provider_called=False,
        provider_status=SearchFallbackProviderStatus.NOT_CALLED,
        deterministic_unsupported_requirements=tuple(intent.unsupported_requirements),
    )


def _invoke_provider(
    provider: SearchFallbackProvider,
    original_text: str,
) -> tuple[
    SearchFallbackProviderResponse | None,
    SearchFallbackProviderStatus | None,
    SearchFallbackShadowRejectionReason | None,
]:
    try:
        response = provider.propose_search(
            request=SearchFallbackProviderRequest(text=original_text)
        )
        if response is None:
            return (
                None,
                SearchFallbackProviderStatus.NO_PROPOSAL,
                SearchFallbackShadowRejectionReason.NO_PROPOSAL,
            )
        return response, None, None
    except SearchFallbackProviderTimeoutError:
        return (
            None,
            SearchFallbackProviderStatus.TIMEOUT,
            SearchFallbackShadowRejectionReason.PROVIDER_TIMEOUT,
        )
    except SearchFallbackProviderError:
        return (
            None,
            SearchFallbackProviderStatus.ERROR,
            SearchFallbackShadowRejectionReason.PROVIDER_ERROR,
        )
    except Exception:
        # Shadow providers must not affect deterministic search availability.
        return (
            None,
            SearchFallbackProviderStatus.ERROR,
            SearchFallbackShadowRejectionReason.PROVIDER_ERROR,
        )


def _evaluate_response(
    shadow_result: SearchFallbackShadowResult,
    response: SearchFallbackProviderResponse,
    original_text: str,
    deterministic_intent: ParsedSearchIntent,
) -> SearchFallbackShadowResult:
    try:
        proposal = UntrustedSearchProposal.model_validate_json(response.proposal_json)
    except ValidationError as exc:
        return _schema_failure(shadow_result, exc)

    try:
        validated_intent = validate_search_proposal(proposal, original_text)
    except SearchProposalValidationError as exc:
        return _proposal_rejection(shadow_result, exc)
    comparison = _compare_semantics(deterministic_intent, validated_intent)
    summary = ValidatedSearchProposalSummary(
        populated_filter_fields=_populated_fields(validated_intent),
        directive_fields=tuple(
            directive.field.value for directive in validated_intent.directives
        ),
    )
    return replace(
        shadow_result,
        provider_called=True,
        provider_status=SearchFallbackProviderStatus.VALIDATED,
        schema_valid=True,
        proposal_valid=True,
        evidence_valid=True,
        exact_filter_match=comparison.exact_filter_match,
        exact_directive_match=comparison.exact_directive_match,
        exact_semantic_match=(
            comparison.exact_filter_match and comparison.exact_directive_match
        ),
        proposal_only_fields=comparison.proposal_only_fields,
        deterministic_only_fields=comparison.deterministic_only_fields,
        disagreed_fields=comparison.disagreed_fields,
        validated_proposal_summary=summary,
    )


def _schema_failure(
    shadow_result: SearchFallbackShadowResult,
    error: ValidationError,
) -> SearchFallbackShadowResult:
    errors = error.errors()
    malformed_json = any(item["type"] == "json_invalid" for item in errors)
    hallucinated_fields = () if malformed_json else _extra_fields(errors)
    return replace(
        shadow_result,
        provider_called=True,
        provider_status=(
            SearchFallbackProviderStatus.MALFORMED_JSON
            if malformed_json
            else SearchFallbackProviderStatus.SCHEMA_INVALID
        ),
        schema_valid=False,
        hallucinated_fields=hallucinated_fields,
        unsupported_invention=bool(hallucinated_fields),
        rejection_reason=(
            SearchFallbackShadowRejectionReason.MALFORMED_JSON
            if malformed_json
            else SearchFallbackShadowRejectionReason.SCHEMA_INVALID
        ),
    )


def _proposal_rejection(
    shadow_result: SearchFallbackShadowResult,
    error: SearchProposalValidationError,
) -> SearchFallbackShadowResult:
    evidence_rejected = error.kind in _EVIDENCE_REJECTIONS
    hallucinated_fields = (
        tuple(field.value for field in error.fields)
        if error.kind in _HALLUCINATION_REJECTIONS
        else ()
    )
    return replace(
        shadow_result,
        provider_called=True,
        provider_status=SearchFallbackProviderStatus.PROPOSAL_REJECTED,
        schema_valid=True,
        proposal_valid=False,
        evidence_valid=False if evidence_rejected else None,
        hallucinated_fields=hallucinated_fields,
        unsupported_invention=(
            error.kind is SearchProposalValidationErrorKind.UNSUPPORTED_CATEGORICAL
        ),
        rejection_reason=error.kind,
    )


def _extra_fields(errors: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(sorted(
        {
            ".".join(str(part) for part in error["loc"])
            for error in errors
            if error["type"] == "extra_forbidden"
        }
    ))


def _compare_semantics(
    deterministic_intent: ParsedSearchIntent,
    proposal_intent: ParsedSearchIntent,
) -> _SemanticComparison:
    deterministic_filters = _filter_values(deterministic_intent)
    proposal_filters = _filter_values(proposal_intent)
    deterministic_fields = set(deterministic_filters)
    proposal_fields = set(proposal_filters)
    return _SemanticComparison(
        exact_filter_match=deterministic_filters == proposal_filters,
        exact_directive_match=(
            _directive_semantics(deterministic_intent)
            == _directive_semantics(proposal_intent)
        ),
        proposal_only_fields=_ordered_fields(proposal_fields - deterministic_fields),
        deterministic_only_fields=_ordered_fields(
            deterministic_fields - proposal_fields
        ),
        disagreed_fields=_ordered_fields(
            {
                field
                for field in deterministic_fields & proposal_fields
                if deterministic_filters[field] != proposal_filters[field]
            }
        ),
    )


def _filter_values(intent: ParsedSearchIntent) -> dict[SearchField, object]:
    return {
        field: getattr(intent.filters, field.value)
        for field in SearchField
        if getattr(intent.filters, field.value) is not None
    }


def _populated_fields(intent: ParsedSearchIntent) -> tuple[str, ...]:
    return tuple(field.value for field in _filter_values(intent))


def _directive_semantics(intent: ParsedSearchIntent) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                directive.field.value,
                directive.strength.value if directive.strength is not None else None,
                directive.preferred,
                directive.allow_alternative,
            )
            for directive in intent.directives
        )
    )


def _ordered_fields(fields: set[SearchField]) -> tuple[str, ...]:
    return tuple(field.value for field in SearchField if field in fields)
