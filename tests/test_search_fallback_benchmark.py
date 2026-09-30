import ast
import inspect
import json

import pytest

from app.ai_engine import search_fallback_benchmark
from app.ai_engine.search_fallback import (
    EvidenceSpan,
    FallbackDisposition,
    ProposedDecimalValue,
    ProposedSearchDirective,
    ProposedSearchFilters,
    ProposedTextValue,
    UntrustedSearchProposal,
)
from app.ai_engine.search_fallback_benchmark import (
    BenchmarkLanguage,
    BenchmarkRate,
    BenchmarkSemanticGroup,
    SearchFallbackBenchmarkCase,
    SearchFallbackBenchmarkMetadata,
    run_search_fallback_benchmark,
    serialize_search_fallback_benchmark_report,
)
from app.ai_engine.search_fallback_shadow import (
    SearchFallbackProviderError,
    SearchFallbackProviderResponse,
    SearchFallbackProviderStatus,
    SearchFallbackProviderTimeoutError,
)
from app.ai_engine.search_strategy import SearchField
from benchmarks.nl_search_golden_cases import (
    GOLDEN_CASES,
    NL_SEARCH_GOLDEN_DATASET_VERSION,
    project_golden_benchmark_cases,
)


class FakeProvider:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.requests = []

    def propose_search(self, *, request):
        self.requests.append(request)
        outcome = self.outcomes.get(request.text)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _metadata():
    return SearchFallbackBenchmarkMetadata(
        report_schema_version="1",
        dataset_version=NL_SEARCH_GOLDEN_DATASET_VERSION,
        provider_name="fake",
        model_identifier="fake-model-snapshot",
        sdk_version="test",
        deterministic_revision="a20a444",
        sampling_configuration="deterministic-fake",
        repetition_id="run-1",
    )


def _case(case_id, query, *, language=BenchmarkLanguage.ENGLISH, group=None):
    return SearchFallbackBenchmarkCase(
        case_id=case_id,
        language=language,
        source_category="test",
        query=query,
        semantic_group=group or BenchmarkSemanticGroup.SUPPORTED_PLUS_UNSUPPORTED,
    )


def _response(proposal):
    return SearchFallbackProviderResponse(proposal.model_dump_json())


def _text_proposal(field, value, text, evidence_text):
    start = text.index(evidence_text)
    return UntrustedSearchProposal(
        filters=ProposedSearchFilters.model_validate({
            field: {
                "value": value,
                "evidence": {"start": start, "end": start + len(evidence_text)},
            }
        })
    )


def test_golden_projection_preserves_every_case_once_in_stable_order():
    projected = project_golden_benchmark_cases()

    assert tuple(case.case_id for case in projected) == tuple(
        case.case_id for case in GOLDEN_CASES
    )
    assert len({case.case_id for case in projected}) == len(projected)
    by_id = {case.case_id: case for case in projected}
    assert by_id["en-basic-search"].language is BenchmarkLanguage.ENGLISH
    assert by_id["ar-basic-search"].language is BenchmarkLanguage.ARABIC
    assert by_id["arz-basic-search"].language is BenchmarkLanguage.ARABIZI
    assert by_id["mixed-ar-en-basic-search"].language is BenchmarkLanguage.MIXED
    assert by_id["blank-query"].language is BenchmarkLanguage.NONE
    assert (
        by_id["en-furnished-only"].semantic_group
        is BenchmarkSemanticGroup.UNSUPPORTED_ONLY
    )
    assert (
        by_id["en-unknown-only"].semantic_group
        is BenchmarkSemanticGroup.UNKNOWN_ONLY
    )
    assert by_id["en-vague-price"].semantic_group is BenchmarkSemanticGroup.VAGUE
    assert (
        by_id["en-negated-location"].semantic_group
        is BenchmarkSemanticGroup.FORBIDDEN
    )
    expected_safety_groups = {
        "safety-company-id": BenchmarkSemanticGroup.FORBIDDEN,
        "safety-tenant": BenchmarkSemanticGroup.FORBIDDEN,
        "safety-scope": BenchmarkSemanticGroup.FORBIDDEN,
        "safety-auth-session": BenchmarkSemanticGroup.FORBIDDEN,
        "safety-price-basis": BenchmarkSemanticGroup.PROMPT_LIKE,
        "safety-prompt-like": BenchmarkSemanticGroup.PROMPT_LIKE,
    }
    assert {
        case_id: by_id[case_id].semantic_group
        for case_id in expected_safety_groups
    } == expected_safety_groups


def test_blank_input_gate_is_forbidden_without_provider_invocation():
    provider = FakeProvider()

    report = run_search_fallback_benchmark(
        cases=(
            _case("blank", "   ", group=BenchmarkSemanticGroup.FORBIDDEN),
        ),
        provider=provider,
        metadata=_metadata(),
    )

    outcome = report.outcomes[0]
    assert outcome.disposition is FallbackDisposition.FORBIDDEN
    assert outcome.provider_called is False
    assert outcome.provider_status is SearchFallbackProviderStatus.NOT_CALLED
    assert provider.requests == []


def test_nonblank_input_uses_interpreter_and_propagates_parser_errors(monkeypatch):
    error = ValueError("nonblank parser error")

    def reject_nonblank_query(text):
        raise error

    monkeypatch.setattr(
        search_fallback_benchmark,
        "interpret_multilingual_search_query",
        reject_nonblank_query,
    )

    with pytest.raises(ValueError) as raised:
        run_search_fallback_benchmark(
            cases=(_case("nonblank", "apartment"),),
            provider=FakeProvider(),
            metadata=_metadata(),
        )

    assert raised.value is error


def test_runner_reports_exact_agreement_and_directive_disagreement():
    exact_text = "furnished apartment"
    exact = _text_proposal(
        "property_type",
        "Apartment",
        exact_text,
        "apartment",
    )
    directive_text = "furnished New Cairo preferred"
    no_directive = _text_proposal(
        "location",
        "New Cairo",
        directive_text,
        "New Cairo",
    )
    provider = FakeProvider({
        exact_text: _response(exact),
        directive_text: _response(no_directive),
    })

    report = run_search_fallback_benchmark(
        cases=(
            _case("exact", exact_text),
            _case("directive-disagreement", directive_text),
        ),
        provider=provider,
        metadata=_metadata(),
    )

    exact_outcome, directive_outcome = report.outcomes
    assert exact_outcome.exact_filter_match is True
    assert exact_outcome.exact_directive_match is True
    assert exact_outcome.exact_semantic_match is True
    assert directive_outcome.exact_filter_match is True
    assert directive_outcome.exact_directive_match is False
    assert directive_outcome.exact_semantic_match is False
    assert len(provider.requests) == 2


def test_runner_separates_proposal_only_and_deterministic_only_fields():
    text = "furnished apartment minimum 8 years"
    property_start = text.index("apartment")
    price_start = text.index("minimum")
    proposal = UntrustedSearchProposal(
        filters=ProposedSearchFilters(
            property_type=ProposedTextValue(
                value="Apartment",
                evidence=EvidenceSpan(
                    start=property_start,
                    end=property_start + len("apartment"),
                ),
            ),
            min_price=ProposedDecimalValue(
                value="8",
                evidence=EvidenceSpan(
                    start=price_start,
                    end=price_start + len("minimum 8"),
                ),
            ),
        )
    )

    report = run_search_fallback_benchmark(
        cases=(_case("field-sets", text),),
        provider=FakeProvider({text: _response(proposal)}),
        metadata=_metadata(),
    )

    outcome = report.outcomes[0]
    assert outcome.proposal_only_fields == ("min_price",)
    assert outcome.deterministic_only_fields == ("min_installment_years",)
    assert outcome.hallucinated_fields == ()


def test_runner_preserves_filter_disagreement_from_shadow_evaluator():
    text = "furnished under 200 sqm and under 6 million"
    price_start = text.index("under 200")
    proposal = UntrustedSearchProposal(
        filters=ProposedSearchFilters(
            max_price=ProposedDecimalValue(
                value="200",
                evidence=EvidenceSpan(
                    start=price_start,
                    end=price_start + len("under 200"),
                ),
            )
        )
    )

    report = run_search_fallback_benchmark(
        cases=(_case("filter-disagreement", text),),
        provider=FakeProvider({text: _response(proposal)}),
        metadata=_metadata(),
    )

    assert report.outcomes[0].disagreed_fields == ("max_price",)
    assert report.outcomes[0].deterministic_only_fields == ("max_area_sqm",)
    assert report.overall.exact_filter_agreement == BenchmarkRate(0, 1)


def test_aggregation_uses_the_frozen_denominators():
    exact_text = "furnished apartment"
    malformed_text = "furnished villa"
    schema_invalid_text = "furnished townhouse"
    evidence_text = "furnished Sheikh Zayed"
    unsupported_text = "furnished Gardenia"
    timeout_text = "furnished sea view"
    error_text = "furnished near school"
    none_text = "furnished"
    exact = _text_proposal(
        "property_type",
        "Apartment",
        exact_text,
        "apartment",
    )
    evidence_mismatch = _text_proposal(
        "location",
        "New Cairo",
        evidence_text,
        "Sheikh Zayed",
    )
    unsupported = _text_proposal(
        "location",
        "Gardenia",
        unsupported_text,
        "Gardenia",
    )
    provider = FakeProvider({
        exact_text: _response(exact),
        malformed_text: SearchFallbackProviderResponse(b"{"),
        schema_invalid_text: SearchFallbackProviderResponse(
            b'{"filters":{},"company_id":7}'
        ),
        evidence_text: _response(evidence_mismatch),
        unsupported_text: _response(unsupported),
        timeout_text: SearchFallbackProviderTimeoutError("private timeout"),
        error_text: SearchFallbackProviderError("private error"),
        none_text: None,
    })
    cases = (
        _case("exact", exact_text),
        _case("malformed", malformed_text),
        _case("schema-invalid", schema_invalid_text),
        _case("evidence", evidence_text),
        _case("unsupported", unsupported_text),
        _case("timeout", timeout_text),
        _case("error", error_text),
        _case("none", none_text),
        _case("skip", "apartment", group=BenchmarkSemanticGroup.SUPPORTED),
        _case("forbidden", "   ", group=BenchmarkSemanticGroup.FORBIDDEN),
    )

    summary = run_search_fallback_benchmark(
        cases=cases,
        provider=provider,
        metadata=_metadata(),
    ).overall

    assert summary.invocation == BenchmarkRate(8, 8)
    assert summary.ineligible_no_call_compliance == BenchmarkRate(2, 2)
    assert summary.schema_valid == BenchmarkRate(3, 5)
    assert summary.proposal_valid == BenchmarkRate(1, 3)
    assert summary.evidence_valid == BenchmarkRate(1, 2)
    assert summary.exact_filter_agreement == BenchmarkRate(1, 1)
    assert summary.exact_directive_agreement == BenchmarkRate(1, 1)
    assert summary.exact_semantic_agreement == BenchmarkRate(1, 1)
    assert summary.timeout == BenchmarkRate(1, 8)
    assert summary.error == BenchmarkRate(1, 8)
    assert summary.no_proposal == BenchmarkRate(1, 8)
    assert summary.hallucination == BenchmarkRate(3, 5)
    assert summary.unsupported_invention == BenchmarkRate(2, 5)
    assert summary.schema_valid_proposal_rejected_without_evidence_result == 1
    assert len(provider.requests) == 8


def test_group_summaries_are_deterministically_ordered():
    report = run_search_fallback_benchmark(
        cases=(
            _case(
                "arabic",
                "شقة",
                language=BenchmarkLanguage.ARABIC,
                group=BenchmarkSemanticGroup.SUPPORTED,
            ),
            SearchFallbackBenchmarkCase(
                case_id="english",
                language=BenchmarkLanguage.ENGLISH,
                source_category="alpha",
                query="apartment",
                semantic_group=BenchmarkSemanticGroup.SUPPORTED,
            ),
        ),
        provider=FakeProvider(),
        metadata=_metadata(),
    )

    assert tuple(group for group, _ in report.by_language) == (
        BenchmarkLanguage.ENGLISH,
        BenchmarkLanguage.ARABIC,
    )
    assert tuple(group for group, _ in report.by_source_category) == (
        "alpha",
        "test",
    )
    assert tuple(group for group, _ in report.by_semantic_group) == (
        BenchmarkSemanticGroup.SUPPORTED,
    )


def test_empty_benchmark_rates_are_undefined():
    report = run_search_fallback_benchmark(
        cases=(),
        provider=FakeProvider(),
        metadata=_metadata(),
    )

    rates = (
        value
        for value in vars(report.overall).values()
        if isinstance(value, BenchmarkRate)
    )
    assert all(rate.rate is None for rate in rates)
    assert report.by_language == ()
    assert report.by_source_category == ()
    assert report.by_semantic_group == ()


def test_duplicate_ids_are_rejected_before_provider_invocation():
    provider = FakeProvider()
    cases = (
        _case("duplicate", "furnished"),
        _case("duplicate", "furnished apartment"),
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        run_search_fallback_benchmark(
            cases=cases,
            provider=provider,
            metadata=_metadata(),
        )

    assert provider.requests == []


def test_serialization_is_deterministic_and_privacy_safe():
    query = "furnished apartment secret-query-marker"
    error_query = "furnished near school private-error-query-marker"
    proposal = _text_proposal(
        "property_type",
        "Apartment",
        query,
        "apartment",
    )
    cases = (
        _case("privacy", query),
        _case("private-error", error_query),
    )
    metadata = _metadata()
    outcomes = {
        query: _response(proposal),
        error_query: SearchFallbackProviderError("secret-exception-marker"),
    }

    first = run_search_fallback_benchmark(
        cases=cases,
        provider=FakeProvider(outcomes),
        metadata=metadata,
    )
    second = run_search_fallback_benchmark(
        cases=cases,
        provider=FakeProvider(outcomes),
        metadata=metadata,
    )
    serialized = serialize_search_fallback_benchmark_report(first)

    assert first == second
    assert serialized == serialize_search_fallback_benchmark_report(second)
    assert query not in serialized
    assert "secret-query-marker" not in serialized
    assert "private-error-query-marker" not in serialized
    assert "secret-exception-marker" not in serialized
    assert "Apartment" not in serialized
    assert '"start"' not in serialized
    assert '"end"' not in serialized
    assert "proposal_json" not in serialized
    assert json.loads(serialized)["outcomes"][0]["case_id"] == "privacy"


def test_runner_reuses_shadow_evaluator_without_execution_imports():
    source = inspect.getsource(search_fallback_benchmark)
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "evaluate_search_fallback_shadow(" in source
    assert not imports & {
        "app.ai_engine.search_application",
        "app.ai_engine.search_orchestrator",
        "app.ai_engine.search_plan_builder",
        "app.auth",
        "app.api",
        "app.search.repository",
        "app.search.service",
    }
