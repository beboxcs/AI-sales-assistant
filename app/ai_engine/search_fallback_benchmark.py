import json
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from typing import Callable, Iterable, TypeVar

from app.ai_engine.multilingual_search_interpreter import (
    interpret_multilingual_search_query,
)
from app.ai_engine.search_fallback import (
    FallbackDisposition,
    SearchProposalValidationErrorKind,
)
from app.ai_engine.search_fallback_shadow import (
    SearchFallbackProvider,
    SearchFallbackProviderStatus,
    SearchFallbackShadowRejectionReason,
    evaluate_search_fallback_shadow,
)
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent


_Group = TypeVar("_Group", bound=Enum)


class BenchmarkLanguage(str, Enum):
    ENGLISH = "english"
    ARABIC = "arabic"
    ARABIZI = "arabizi"
    MIXED = "mixed"
    NONE = "none"


class BenchmarkSemanticGroup(str, Enum):
    SUPPORTED = "supported"
    SUPPORTED_PLUS_UNSUPPORTED = "supported_plus_unsupported"
    UNSUPPORTED_ONLY = "unsupported_only"
    UNKNOWN_ONLY = "unknown_only"
    VAGUE = "vague"
    FORBIDDEN = "forbidden"
    PROMPT_LIKE = "prompt_like"


@dataclass(frozen=True)
class SearchFallbackBenchmarkCase:
    case_id: str
    language: BenchmarkLanguage
    source_category: str
    query: str
    semantic_group: BenchmarkSemanticGroup
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchFallbackBenchmarkRunConfiguration:
    timeout_seconds: int
    max_retries: int
    context_window_tokens: int
    max_input_tokens_per_call: int
    max_output_tokens: int
    repetitions: int
    max_provider_calls: int
    local_budget_cap_usd: str
    input_cost_per_million_tokens_usd: str
    output_cost_per_million_tokens_usd: str
    estimated_worst_case_cost_usd: str


@dataclass(frozen=True)
class SearchFallbackBenchmarkMetadata:
    report_schema_version: str
    dataset_version: str
    provider_name: str
    model_identifier: str
    sdk_version: str
    deterministic_revision: str
    sampling_configuration: str
    repetition_id: str
    caller_supplied_timestamp: str | None = None
    run_configuration: SearchFallbackBenchmarkRunConfiguration | None = None


@dataclass(frozen=True)
class SearchFallbackBenchmarkCaseOutcome:
    case_id: str
    language: BenchmarkLanguage
    source_category: str
    semantic_group: BenchmarkSemanticGroup
    tags: tuple[str, ...]
    disposition: FallbackDisposition
    provider_called: bool
    provider_status: SearchFallbackProviderStatus
    schema_valid: bool | None
    proposal_valid: bool | None
    evidence_valid: bool | None
    exact_filter_match: bool | None
    exact_directive_match: bool | None
    exact_semantic_match: bool | None
    proposal_only_fields: tuple[str, ...]
    deterministic_only_fields: tuple[str, ...]
    disagreed_fields: tuple[str, ...]
    hallucinated_fields: tuple[str, ...]
    unsupported_invention: bool
    rejection_reason: (
        SearchFallbackShadowRejectionReason
        | SearchProposalValidationErrorKind
        | None
    )


@dataclass(frozen=True)
class BenchmarkRate:
    numerator: int
    denominator: int

    @property
    def rate(self) -> Fraction | None:
        if self.denominator == 0:
            return None
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class SearchFallbackBenchmarkSummary:
    case_count: int
    invocation: BenchmarkRate
    ineligible_no_call_compliance: BenchmarkRate
    schema_valid: BenchmarkRate
    proposal_valid: BenchmarkRate
    evidence_valid: BenchmarkRate
    exact_filter_agreement: BenchmarkRate
    exact_directive_agreement: BenchmarkRate
    exact_semantic_agreement: BenchmarkRate
    timeout: BenchmarkRate
    error: BenchmarkRate
    no_proposal: BenchmarkRate
    hallucination: BenchmarkRate
    unsupported_invention: BenchmarkRate
    schema_valid_proposal_rejected_without_evidence_result: int


@dataclass(frozen=True)
class SearchFallbackBenchmarkReport:
    metadata: SearchFallbackBenchmarkMetadata
    outcomes: tuple[SearchFallbackBenchmarkCaseOutcome, ...]
    overall: SearchFallbackBenchmarkSummary
    by_language: tuple[tuple[BenchmarkLanguage, SearchFallbackBenchmarkSummary], ...]
    by_source_category: tuple[tuple[str, SearchFallbackBenchmarkSummary], ...]
    by_semantic_group: tuple[
        tuple[BenchmarkSemanticGroup, SearchFallbackBenchmarkSummary], ...
    ]


def run_search_fallback_benchmark(
    *,
    cases: Iterable[SearchFallbackBenchmarkCase],
    provider: SearchFallbackProvider,
    metadata: SearchFallbackBenchmarkMetadata,
) -> SearchFallbackBenchmarkReport:
    benchmark_cases = tuple(cases)
    _validate_unique_case_ids(benchmark_cases)

    outcomes = tuple(
        _evaluate_case(case, provider)
        for case in benchmark_cases
    )
    return SearchFallbackBenchmarkReport(
        metadata=metadata,
        outcomes=outcomes,
        overall=_summarize(outcomes),
        by_language=_group_summaries(
            outcomes,
            BenchmarkLanguage,
            lambda outcome: outcome.language,
        ),
        by_source_category=tuple(
            (
                category,
                _summarize(tuple(
                    outcome
                    for outcome in outcomes
                    if outcome.source_category == category
                )),
            )
            for category in sorted({outcome.source_category for outcome in outcomes})
        ),
        by_semantic_group=_group_summaries(
            outcomes,
            BenchmarkSemanticGroup,
            lambda outcome: outcome.semantic_group,
        ),
    )


def serialize_search_fallback_benchmark_report(
    report: SearchFallbackBenchmarkReport,
) -> str:
    return json.dumps(asdict(report), ensure_ascii=True, indent=2, sort_keys=True)


def _validate_unique_case_ids(cases: tuple[SearchFallbackBenchmarkCase, ...]) -> None:
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case IDs must be unique")


def _evaluate_case(
    case: SearchFallbackBenchmarkCase,
    provider: SearchFallbackProvider,
) -> SearchFallbackBenchmarkCaseOutcome:
    intent = (
        ParsedSearchIntent(filters=SearchFilters())
        if not case.query.strip()
        else interpret_multilingual_search_query(case.query)
    )
    shadow = evaluate_search_fallback_shadow(
        original_text=case.query,
        deterministic_intent=intent,
        provider=provider,
    )
    return SearchFallbackBenchmarkCaseOutcome(
        case_id=case.case_id,
        language=case.language,
        source_category=case.source_category,
        semantic_group=case.semantic_group,
        tags=case.tags,
        disposition=shadow.disposition,
        provider_called=shadow.provider_called,
        provider_status=shadow.provider_status,
        schema_valid=shadow.schema_valid,
        proposal_valid=shadow.proposal_valid,
        evidence_valid=shadow.evidence_valid,
        exact_filter_match=shadow.exact_filter_match,
        exact_directive_match=shadow.exact_directive_match,
        exact_semantic_match=shadow.exact_semantic_match,
        proposal_only_fields=shadow.proposal_only_fields,
        deterministic_only_fields=shadow.deterministic_only_fields,
        disagreed_fields=shadow.disagreed_fields,
        hallucinated_fields=shadow.hallucinated_fields,
        unsupported_invention=shadow.unsupported_invention,
        rejection_reason=shadow.rejection_reason,
    )


def _summarize(
    outcomes: tuple[SearchFallbackBenchmarkCaseOutcome, ...],
) -> SearchFallbackBenchmarkSummary:
    eligible = lambda outcome: outcome.disposition in {
        FallbackDisposition.SHADOW_ONLY,
        FallbackDisposition.PROPOSAL_ELIGIBLE,
    }
    ineligible = lambda outcome: outcome.disposition in {
        FallbackDisposition.SKIP,
        FallbackDisposition.FORBIDDEN,
    }
    schema_reached = lambda outcome: outcome.schema_valid is not None
    schema_valid = lambda outcome: outcome.schema_valid is True
    evidence_reached = lambda outcome: outcome.evidence_valid is not None
    proposal_valid = lambda outcome: outcome.proposal_valid is True
    provider_called = lambda outcome: outcome.provider_called

    return SearchFallbackBenchmarkSummary(
        case_count=len(outcomes),
        invocation=_rate(outcomes, eligible, lambda outcome: outcome.provider_called),
        ineligible_no_call_compliance=_rate(
            outcomes,
            ineligible,
            lambda outcome: not outcome.provider_called,
        ),
        schema_valid=_rate(outcomes, schema_reached, schema_valid),
        proposal_valid=_rate(outcomes, schema_valid, proposal_valid),
        evidence_valid=_rate(
            outcomes,
            evidence_reached,
            lambda outcome: outcome.evidence_valid is True,
        ),
        exact_filter_agreement=_rate(
            outcomes,
            proposal_valid,
            lambda outcome: outcome.exact_filter_match is True,
        ),
        exact_directive_agreement=_rate(
            outcomes,
            proposal_valid,
            lambda outcome: outcome.exact_directive_match is True,
        ),
        exact_semantic_agreement=_rate(
            outcomes,
            proposal_valid,
            lambda outcome: outcome.exact_semantic_match is True,
        ),
        timeout=_status_rate(
            outcomes,
            provider_called,
            SearchFallbackProviderStatus.TIMEOUT,
        ),
        error=_status_rate(
            outcomes,
            provider_called,
            SearchFallbackProviderStatus.ERROR,
        ),
        no_proposal=_status_rate(
            outcomes,
            provider_called,
            SearchFallbackProviderStatus.NO_PROPOSAL,
        ),
        hallucination=_rate(
            outcomes,
            schema_reached,
            lambda outcome: bool(outcome.hallucinated_fields),
        ),
        unsupported_invention=_rate(
            outcomes,
            schema_reached,
            lambda outcome: outcome.unsupported_invention,
        ),
        schema_valid_proposal_rejected_without_evidence_result=sum(
            outcome.schema_valid is True
            and outcome.proposal_valid is False
            and outcome.evidence_valid is None
            for outcome in outcomes
        ),
    )


def _rate(
    outcomes: tuple[SearchFallbackBenchmarkCaseOutcome, ...],
    denominator: Callable[[SearchFallbackBenchmarkCaseOutcome], bool],
    numerator: Callable[[SearchFallbackBenchmarkCaseOutcome], bool],
) -> BenchmarkRate:
    denominator_count = sum(denominator(outcome) for outcome in outcomes)
    numerator_count = sum(
        denominator(outcome) and numerator(outcome)
        for outcome in outcomes
    )
    return BenchmarkRate(numerator_count, denominator_count)


def _status_rate(
    outcomes: tuple[SearchFallbackBenchmarkCaseOutcome, ...],
    denominator: Callable[[SearchFallbackBenchmarkCaseOutcome], bool],
    status: SearchFallbackProviderStatus,
) -> BenchmarkRate:
    return _rate(
        outcomes,
        denominator,
        lambda outcome: outcome.provider_status is status,
    )


def _group_summaries(
    outcomes: tuple[SearchFallbackBenchmarkCaseOutcome, ...],
    enum_type: type[_Group],
    key: Callable[[SearchFallbackBenchmarkCaseOutcome], _Group],
) -> tuple[tuple[_Group, SearchFallbackBenchmarkSummary], ...]:
    return tuple(
        (
            group,
            _summarize(tuple(outcome for outcome in outcomes if key(outcome) is group)),
        )
        for group in enum_type
        if any(key(outcome) is group for outcome in outcomes)
    )
