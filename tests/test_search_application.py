import inspect

import pytest

from app.ai_engine import search_application
from app.ai_engine.search_filters import SearchFilters
from app.search.contracts import SearchExecutionContext


def test_search_units_composes_policy_planner_and_orchestrator(monkeypatch):
    filters = SearchFilters(location="New Cairo")
    context = SearchExecutionContext(company_id=7, resolved_scope_keys=())
    strategy = object()
    plan = object()
    expected_result = object()
    calls = []

    def build_strategy(received_filters):
        calls.append(("strategy", received_filters))
        return strategy

    def build_plan(received_strategy):
        calls.append(("plan", received_strategy))
        return plan

    def execute(*, strategy, plan, context):
        calls.append(("orchestrator", strategy, plan, context))
        return expected_result

    monkeypatch.setattr(
        search_application,
        "build_default_search_strategy",
        build_strategy,
    )
    monkeypatch.setattr(search_application, "build_search_plan", build_plan)
    monkeypatch.setattr(search_application, "execute_search_intelligence", execute)

    result = search_application.search_units(filters=filters, context=context)

    assert result is expected_result
    assert calls == [
        ("strategy", filters),
        ("plan", strategy),
        ("orchestrator", strategy, plan, context),
    ]


def test_search_units_passes_empty_scopes_unchanged(monkeypatch):
    filters = SearchFilters()
    context = SearchExecutionContext(company_id=7, resolved_scope_keys=())
    captured = {}

    monkeypatch.setattr(
        search_application,
        "build_default_search_strategy",
        lambda received_filters: received_filters,
    )
    monkeypatch.setattr(
        search_application,
        "build_search_plan",
        lambda strategy: strategy,
    )

    expected = object()

    def execute(*, strategy, plan, context):
        captured.update(strategy=strategy, plan=plan, context=context)
        return expected

    monkeypatch.setattr(search_application, "execute_search_intelligence", execute)

    result = search_application.search_units(filters=filters, context=context)

    assert result is expected
    assert captured["context"] is context
    assert captured["context"].resolved_scope_keys == ()


def test_search_units_propagates_strategy_builder_errors(monkeypatch):
    expected_error = ValueError("strategy policy failed")

    def fail(_):
        raise expected_error

    monkeypatch.setattr(search_application, "build_default_search_strategy", fail)

    with pytest.raises(ValueError) as raised:
        search_application.search_units(
            filters=SearchFilters(),
            context=SearchExecutionContext(7, ("scope:one",)),
        )

    assert raised.value is expected_error


def test_search_units_propagates_planner_errors(monkeypatch):
    expected_error = RuntimeError("planner failed")
    strategy = object()

    monkeypatch.setattr(
        search_application,
        "build_default_search_strategy",
        lambda _: strategy,
    )

    def fail(_):
        raise expected_error

    monkeypatch.setattr(search_application, "build_search_plan", fail)

    with pytest.raises(RuntimeError) as raised:
        search_application.search_units(
            filters=SearchFilters(),
            context=SearchExecutionContext(7, ("scope:one",)),
        )

    assert raised.value is expected_error


def test_search_units_propagates_orchestrator_errors(monkeypatch):
    expected_error = RuntimeError("orchestrator failed")
    strategy = object()
    plan = object()
    monkeypatch.setattr(
        search_application,
        "build_default_search_strategy",
        lambda _: strategy,
    )
    monkeypatch.setattr(search_application, "build_search_plan", lambda _: plan)

    def fail(**_):
        raise expected_error

    monkeypatch.setattr(search_application, "execute_search_intelligence", fail)

    with pytest.raises(RuntimeError) as raised:
        search_application.search_units(
            filters=SearchFilters(),
            context=SearchExecutionContext(7, ("scope:one",)),
        )

    assert raised.value is expected_error


def test_search_units_does_not_mutate_filters(monkeypatch):
    filters = SearchFilters(location="New Cairo", max_price=8_000_000)
    before = filters.model_dump()
    strategy = object()
    plan = object()

    def build_strategy(received_filters):
        assert received_filters is filters
        return strategy

    monkeypatch.setattr(
        search_application,
        "build_default_search_strategy",
        build_strategy,
    )
    monkeypatch.setattr(search_application, "build_search_plan", lambda _: plan)
    monkeypatch.setattr(
        search_application,
        "execute_search_intelligence",
        lambda **_: object(),
    )

    search_application.search_units(
        filters=filters,
        context=SearchExecutionContext(7, ("scope:one",)),
    )

    assert filters.model_dump() == before


def test_search_units_repeated_calls_preserve_boundary_behavior(monkeypatch):
    filters = SearchFilters(location="New Cairo")
    context = SearchExecutionContext(7, ("scope:one",))
    strategy = object()
    plan = object()
    expected = object()
    monkeypatch.setattr(
        search_application,
        "build_default_search_strategy",
        lambda _: strategy,
    )
    monkeypatch.setattr(search_application, "build_search_plan", lambda _: plan)
    monkeypatch.setattr(
        search_application,
        "execute_search_intelligence",
        lambda **_: expected,
    )

    first = search_application.search_units(filters=filters, context=context)
    second = search_application.search_units(filters=filters, context=context)

    assert first is expected
    assert second is expected


def test_facade_has_no_database_or_api_imports():
    source = inspect.getsource(search_application)

    assert "sqlalchemy" not in source.lower()
    assert "fastapi" not in source.lower()
    assert "search.repository" not in source
    assert "SessionLocal" not in source
