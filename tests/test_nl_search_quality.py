from decimal import Decimal

import pytest

from benchmarks.nl_search_golden_cases import (
    DirectiveExpectation,
    GOLDEN_CASES,
)
from app.ai_engine.search_application import search_units_from_query
from app.ai_engine.search_fallback import decide_search_fallback
from app.ai_engine.search_strategy import SearchField
from app.search import service as search_service
from app.search.contracts import PriceBasis, SearchExecutionContext


@pytest.fixture
def evaluate_query(monkeypatch):
    context = SearchExecutionContext(42, ("scope:trusted",))
    calls = []

    def execute_search(*, filters, context, price_basis):
        calls.append((filters, context, price_basis))
        return []

    monkeypatch.setattr(search_service, "execute_search", execute_search)

    def evaluate(text):
        calls.clear()
        result = search_units_from_query(text=text, context=context)
        return result, tuple(calls), context

    return evaluate


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.case_id)
def test_golden_query_semantics(case, evaluate_query):
    result, calls, context = evaluate_query(case.query)
    mismatches = _semantic_mismatches(case, result)

    if bool(calls) is not case.execute:
        mismatches.append(_mismatch(case, "backend execution", case.execute, bool(calls)))
    if any(call_context is not context for _, call_context, _ in calls):
        mismatches.append(
            _mismatch(case, "trusted context identity", "original object", "different object")
        )
    if case.expected_price_basis is not None:
        actual_bases = {price_basis for _, _, price_basis in calls}
        expected_bases = {case.expected_price_basis}
        if actual_bases != expected_bases:
            mismatches.append(
                _mismatch(case, "PriceBasis", expected_bases, actual_bases)
            )
    if case.expected_fallback_disposition is not None:
        decision = decide_search_fallback(result.intent, case.query)
        if decision.disposition is not case.expected_fallback_disposition:
            mismatches.append(_mismatch(
                case,
                "fallback disposition",
                case.expected_fallback_disposition,
                decision.disposition,
            ))
        if decision.reason is not case.expected_fallback_reason:
            mismatches.append(_mismatch(
                case,
                "fallback reason",
                case.expected_fallback_reason,
                decision.reason,
            ))

    if mismatches:
        pytest.fail("\n".join(mismatches))


@pytest.mark.parametrize(
    "group",
    sorted({case.equivalence_group for case in GOLDEN_CASES if case.equivalence_group}),
)
def test_multilingual_equivalence_groups(group, evaluate_query):
    cases = [case for case in GOLDEN_CASES if case.equivalence_group == group]
    observed = [(case, evaluate_query(case.query)[0].intent) for case in cases]
    members = tuple((case.case_id, case.query) for case, _ in observed)

    for index, (left_case, left_intent) in enumerate(observed):
        for right_case, right_intent in observed[index + 1 :]:
            assert left_intent == right_intent, (
                f"equivalence_group={group!r}; members={members!r}; "
                f"unequal cases={left_case.case_id!r}, {right_case.case_id!r}; "
                f"left intent={left_intent!r}; right intent={right_intent!r}"
            )


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.case_id)
def test_golden_query_evaluation_is_deterministic(case, evaluate_query):
    first_result, first_calls, _ = evaluate_query(case.query)
    second_result, second_calls, _ = evaluate_query(case.query)

    assert first_result.intent == second_result.intent, _mismatch(
        case,
        "repeated intent",
        first_result.intent,
        second_result.intent,
    )
    assert (first_result.search is not None) is (second_result.search is not None), (
        _mismatch(
            case,
            "repeated execution eligibility",
            first_result.search is not None,
            second_result.search is not None,
        )
    )
    assert first_calls == second_calls, _mismatch(
        case,
        "repeated backend calls",
        first_calls,
        second_calls,
    )


def _semantic_mismatches(case, result):
    mismatches = []
    expected_filters = dict(case.expected_filters)
    for field in SearchField:
        expected = expected_filters.get(field)
        actual = getattr(result.intent.filters, field.value)
        if actual != expected or (
            isinstance(expected, Decimal) and not isinstance(actual, Decimal)
        ):
            mismatches.append(_mismatch(case, f"filter.{field.value}", expected, actual))

    actual_directives = tuple(
        DirectiveExpectation(
            field=directive.field,
            strength=directive.strength,
            preferred=directive.preferred,
            allow_alternative=directive.allow_alternative,
        )
        for directive in result.intent.directives
    )
    if actual_directives != case.expected_directives:
        mismatches.append(
            _mismatch(
                case,
                "directives",
                case.expected_directives,
                actual_directives,
            )
        )

    actual_ambiguities = result.intent.ambiguities
    if len(actual_ambiguities) != len(case.expected_ambiguities):
        mismatches.append(
            _mismatch(
                case,
                "ambiguity count",
                len(case.expected_ambiguities),
                len(actual_ambiguities),
            )
        )
    for index, expected in enumerate(case.expected_ambiguities):
        if index >= len(actual_ambiguities):
            break
        actual = actual_ambiguities[index]
        if actual.field != expected.field:
            mismatches.append(
                _mismatch(
                    case,
                    f"ambiguities[{index}].field",
                    expected.field,
                    actual.field,
                )
            )
        if expected.reason_contains not in actual.reason:
            mismatches.append(
                _mismatch(
                    case,
                    f"ambiguities[{index}].reason",
                    f"contains {expected.reason_contains!r}",
                    actual.reason,
                )
            )

    actual_unsupported = tuple(result.intent.unsupported_requirements)
    if actual_unsupported != case.expected_unsupported:
        mismatches.append(
            _mismatch(
                case,
                "unsupported requirements",
                case.expected_unsupported,
                actual_unsupported,
            )
        )

    actual_execute = result.search is not None
    if actual_execute is not case.execute:
        mismatches.append(
            _mismatch(case, "execution eligibility", case.execute, actual_execute)
        )
    return mismatches


def _mismatch(case, dimension, expected, actual):
    return (
        f"case_id={case.case_id!r}; query={case.query!r}; "
        f"dimension={dimension}; expected={expected!r}; actual={actual!r}"
    )
