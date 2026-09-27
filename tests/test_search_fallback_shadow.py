from dataclasses import FrozenInstanceError

import pytest

from app.ai_engine.multilingual_search_interpreter import (
    interpret_multilingual_search_query,
)
from app.ai_engine.search_fallback import (
    EvidenceSpan,
    FallbackDisposition,
    ProposedDecimalValue,
    ProposedSearchDirective,
    ProposedSearchFilters,
    ProposedTextValue,
    SearchProposalValidationErrorKind,
    UntrustedSearchProposal,
)
from app.ai_engine.search_fallback_shadow import (
    SearchFallbackProviderError,
    SearchFallbackProviderRequest,
    SearchFallbackProviderResponse,
    SearchFallbackProviderStatus,
    SearchFallbackProviderTimeoutError,
    SearchFallbackShadowRejectionReason,
    evaluate_search_fallback_shadow,
)
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_strategy import SearchField


class FakeProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def propose_search(self, *, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def _intent(*, filters=None, unsupported=()):
    return ParsedSearchIntent(
        filters=filters or SearchFilters(),
        unsupported_requirements=list(unsupported),
    )


def _response(proposal):
    return SearchFallbackProviderResponse(proposal_json=proposal.model_dump_json())


def _text_proposal(field, value, text, *, start=0, end=None):
    evidence = EvidenceSpan(start=start, end=len(text) if end is None else end)
    return UntrustedSearchProposal(
        filters=ProposedSearchFilters.model_validate({
            field: {"value": value, "evidence": evidence.model_dump()}
        })
    )


@pytest.mark.parametrize(
    ("text", "intent", "disposition"),
    [
        (
            "apartment",
            _intent(filters=SearchFilters(property_type="Apartment")),
            FallbackDisposition.SKIP,
        ),
        ("search tenant 3", _intent(), FallbackDisposition.FORBIDDEN),
    ],
)
def test_ineligible_decisions_do_not_call_provider(text, intent, disposition):
    provider = FakeProvider(error=AssertionError("provider must not be called"))

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=intent,
        provider=provider,
    )

    assert shadow.disposition is disposition
    assert shadow.provider_called is False
    assert shadow.provider_status is SearchFallbackProviderStatus.NOT_CALLED
    assert provider.requests == []


@pytest.mark.parametrize(
    ("text", "intent", "disposition"),
    [
        (
            "furnished",
            _intent(unsupported=("furnished",)),
            FallbackDisposition.SHADOW_ONLY,
        ),
        ("Gardenia", _intent(), FallbackDisposition.PROPOSAL_ELIGIBLE),
    ],
)
def test_eligible_decisions_call_provider_once(text, intent, disposition):
    provider = FakeProvider()

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=intent,
        provider=provider,
    )

    assert shadow.disposition is disposition
    assert shadow.provider_status is SearchFallbackProviderStatus.NO_PROPOSAL
    assert provider.requests == [SearchFallbackProviderRequest(text=text)]


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (
            SearchFallbackProviderTimeoutError("private timeout"),
            SearchFallbackProviderStatus.TIMEOUT,
            SearchFallbackShadowRejectionReason.PROVIDER_TIMEOUT,
        ),
        (
            SearchFallbackProviderError("private provider error"),
            SearchFallbackProviderStatus.ERROR,
            SearchFallbackShadowRejectionReason.PROVIDER_ERROR,
        ),
        (
            RuntimeError("private unexpected error"),
            SearchFallbackProviderStatus.ERROR,
            SearchFallbackShadowRejectionReason.PROVIDER_ERROR,
        ),
    ],
)
def test_provider_failures_are_safe_and_typed(error, status, reason):
    shadow = evaluate_search_fallback_shadow(
        original_text="Gardenia",
        deterministic_intent=_intent(),
        provider=FakeProvider(error=error),
    )

    assert shadow.provider_status is status
    assert shadow.rejection_reason is reason
    assert "private" not in repr(shadow)


@pytest.mark.parametrize(
    ("proposal_json", "status", "reason"),
    [
        (
            b"{",
            SearchFallbackProviderStatus.MALFORMED_JSON,
            SearchFallbackShadowRejectionReason.MALFORMED_JSON,
        ),
        (
            b'{"filters":{"location":{"value":"New Cairo"}}}',
            SearchFallbackProviderStatus.SCHEMA_INVALID,
            SearchFallbackShadowRejectionReason.SCHEMA_INVALID,
        ),
    ],
)
def test_invalid_provider_json_is_classified_structurally(
    proposal_json, status, reason
):
    shadow = evaluate_search_fallback_shadow(
        original_text="Gardenia",
        deterministic_intent=_intent(),
        provider=FakeProvider(SearchFallbackProviderResponse(proposal_json)),
    )

    assert shadow.provider_status is status
    assert shadow.schema_valid is False
    assert shadow.rejection_reason is reason


def test_extra_authority_field_is_reported_without_retaining_payload():
    payload = b'{"filters":{},"company_id":7}'

    shadow = evaluate_search_fallback_shadow(
        original_text="Gardenia",
        deterministic_intent=_intent(),
        provider=FakeProvider(SearchFallbackProviderResponse(payload)),
    )

    assert shadow.hallucinated_fields == ("company_id",)
    assert shadow.unsupported_invention is True
    assert payload.decode() not in repr(shadow)


def test_valid_proposal_exactly_matches_deterministic_semantics():
    text = "furnished apartment under 1234567890123456.78; apartment preferred"
    intent = interpret_multilingual_search_query(text)
    price_start = text.index("under")
    property_start = text.rindex("apartment")
    property_evidence = EvidenceSpan(start=property_start, end=len(text))
    price_evidence = EvidenceSpan(start=price_start, end=text.index(";"))
    proposal = UntrustedSearchProposal(
        filters=ProposedSearchFilters(
            property_type=ProposedTextValue(
                value="Apartment",
                evidence=property_evidence,
            ),
            max_price=ProposedDecimalValue(
                value="1234567890123456.78",
                evidence=price_evidence,
            ),
        ),
        directives=(
            ProposedSearchDirective(
                field=SearchField.PROPERTY_TYPE,
                preferred=True,
                evidence=property_evidence,
            ),
        ),
    )

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=intent,
        provider=FakeProvider(_response(proposal)),
    )

    assert shadow.provider_status is SearchFallbackProviderStatus.VALIDATED
    assert shadow.schema_valid is shadow.proposal_valid is shadow.evidence_valid is True
    assert shadow.exact_filter_match is True
    assert shadow.exact_directive_match is True
    assert shadow.exact_semantic_match is True
    assert shadow.validated_proposal_summary.populated_filter_fields == (
        "property_type",
        "max_price",
    )
    assert shadow.validated_proposal_summary.directive_fields == ("property_type",)
    assert "Apartment" not in repr(shadow)
    assert "1234567890123456.78" not in repr(shadow)


def test_valid_proposal_only_field_is_disagreement_not_hallucination():
    text = "furnished apartment"
    start = text.index("apartment")
    proposal = _text_proposal("property_type", "Apartment", text, start=start)

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=_intent(unsupported=("furnished",)),
        provider=FakeProvider(_response(proposal)),
    )

    assert shadow.proposal_only_fields == ("property_type",)
    assert shadow.hallucinated_fields == ()
    assert shadow.unsupported_invention is False


def test_comparison_separates_missing_differing_and_directive_semantics():
    text = "furnished New Cairo preferred; Sheikh Zayed"
    zayed_start = text.index("Sheikh Zayed")
    proposal = _text_proposal("location", "Sheikh Zayed", text, start=zayed_start)
    deterministic = ParsedSearchIntent.model_validate({
        "filters": {"location": "New Cairo", "property_type": "Apartment"},
        "directives": [{"field": "location", "preferred": True}],
        "unsupported_requirements": ["furnished"],
    })

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=deterministic,
        provider=FakeProvider(_response(proposal)),
    )

    assert shadow.deterministic_only_fields == ("property_type",)
    assert shadow.disagreed_fields == ("location",)
    assert shadow.exact_filter_match is False
    assert shadow.exact_directive_match is False
    assert shadow.exact_semantic_match is False


@pytest.mark.parametrize(
    ("text", "proposal", "kind", "hallucinated", "unsupported", "evidence_valid"),
    [
        (
            "furnished Sheikh Zayed",
            _text_proposal("location", "New Cairo", "furnished Sheikh Zayed", start=10),
            SearchProposalValidationErrorKind.EVIDENCE_MISMATCH,
            ("location",),
            False,
            False,
        ),
        (
            "furnished Gardenia",
            _text_proposal("location", "Gardenia", "furnished Gardenia", start=10),
            SearchProposalValidationErrorKind.UNSUPPORTED_CATEGORICAL,
            ("location",),
            True,
            None,
        ),
    ],
)
def test_typed_proposal_rejections_drive_safe_classification(
    text, proposal, kind, hallucinated, unsupported, evidence_valid
):
    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=_intent(unsupported=("furnished",)),
        provider=FakeProvider(_response(proposal)),
    )

    assert shadow.provider_status is SearchFallbackProviderStatus.PROPOSAL_REJECTED
    assert shadow.rejection_reason is kind
    assert shadow.hallucinated_fields == hallucinated
    assert shadow.unsupported_invention is unsupported
    assert shadow.evidence_valid is evidence_valid


def test_contradictory_proposal_bounds_are_rejected_without_hallucination():
    text = "furnished above 8 million under 6 million"
    min_end = text.index(" under")
    proposal = UntrustedSearchProposal(
        filters=ProposedSearchFilters(
            min_price=ProposedDecimalValue(
                value="8000000",
                evidence=EvidenceSpan(start=text.index("above"), end=min_end),
            ),
            max_price=ProposedDecimalValue(
                value="6000000",
                evidence=EvidenceSpan(start=text.index("under"), end=len(text)),
            ),
        )
    )

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=_intent(unsupported=("furnished",)),
        provider=FakeProvider(_response(proposal)),
    )

    assert (
        shadow.rejection_reason
        is SearchProposalValidationErrorKind.CONTRADICTORY_BOUNDS
    )
    assert shadow.hallucinated_fields == ()
    assert shadow.unsupported_invention is False


def test_duplicate_directives_are_typed_invalid_directive_rejection():
    text = "furnished New Cairo preferred"
    start = text.index("New Cairo")
    evidence = EvidenceSpan(start=start, end=len(text))
    directive = ProposedSearchDirective(
        field=SearchField.LOCATION,
        preferred=True,
        evidence=evidence,
    )
    proposal = UntrustedSearchProposal(
        filters=ProposedSearchFilters(
            location=ProposedTextValue(value="New Cairo", evidence=evidence)
        ),
        directives=(directive, directive),
    )

    shadow = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=_intent(unsupported=("furnished",)),
        provider=FakeProvider(_response(proposal)),
    )

    assert (
        shadow.rejection_reason
        is SearchProposalValidationErrorKind.INVALID_DIRECTIVE
    )
    assert shadow.evidence_valid is None


def test_shadow_result_is_immutable_private_and_deterministic():
    text = "furnished Gardenia"
    proposal = _text_proposal("location", "Gardenia", text, start=10)
    provider = FakeProvider(_response(proposal))

    first = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=_intent(unsupported=("furnished",)),
        provider=provider,
    )
    second = evaluate_search_fallback_shadow(
        original_text=text,
        deterministic_intent=_intent(unsupported=("furnished",)),
        provider=provider,
    )

    assert first == second
    assert "Gardenia" not in repr(first)
    with pytest.raises(FrozenInstanceError):
        first.provider_called = False
