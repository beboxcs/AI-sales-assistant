from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.database import engine
from app.models import Company, InventoryScope
from app.search import scope_resolution
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchScopeResolutionFailedError,
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    assert engine.dialect.name == "postgresql"

    connection = engine.connect()
    outer_transaction = connection.begin()
    session = Session(
        bind=connection,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


def test_none_resolves_all_company_scopes_in_id_order(
    db_session: Session,
    monkeypatch,
):
    company_a = _add_company(db_session)
    company_b = _add_company(db_session)
    scope_a_first = _add_scope(db_session, company_a, "scope:a:first")
    scope_a_second = _add_scope(db_session, company_a, "scope:a:second")
    _add_scope(db_session, company_b, "scope:b")
    expected = tuple(
        scope.scope_key
        for scope in sorted(
            (scope_a_first, scope_a_second),
            key=lambda scope: scope.id,
        )
    )

    context = _resolve(
        db_session,
        monkeypatch,
        company_id=company_a.id,
        requested_scope_keys=None,
    )

    assert context.company_id == company_a.id
    assert context.resolved_scope_keys == expected


def test_none_returns_empty_when_company_has_no_scopes(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)

    context = _resolve(
        db_session,
        monkeypatch,
        company_id=company.id,
        requested_scope_keys=None,
    )

    assert context.resolved_scope_keys == ()


def test_explicit_empty_scope_selection_fails_closed_without_query(monkeypatch):
    def fail_if_called():
        raise AssertionError("SessionLocal must not be called")

    monkeypatch.setattr(scope_resolution, "SessionLocal", fail_if_called)

    context = scope_resolution.resolve_search_execution_context(
        company_id=1,
        requested_scope_keys=(),
    )

    assert context.company_id == 1
    assert context.resolved_scope_keys == ()


def test_explicit_scopes_preserve_requested_order(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    first_by_id = _add_scope(db_session, company, "scope:first")
    second_by_id = _add_scope(db_session, company, "scope:second")
    scopes_by_id = sorted((first_by_id, second_by_id), key=lambda scope: scope.id)
    requested = (scopes_by_id[1].scope_key, scopes_by_id[0].scope_key)

    context = _resolve(
        db_session,
        monkeypatch,
        company_id=company.id,
        requested_scope_keys=requested,
    )

    assert context.resolved_scope_keys == requested


def test_duplicate_explicit_scopes_are_rejected_without_query(monkeypatch):
    def fail_if_called():
        raise AssertionError("SessionLocal must not be called")

    monkeypatch.setattr(scope_resolution, "SessionLocal", fail_if_called)

    with pytest.raises(InvalidSearchScopeError):
        scope_resolution.resolve_search_execution_context(
            company_id=1,
            requested_scope_keys=("scope:a", "scope:a"),
        )


def test_unknown_and_cross_tenant_scopes_are_indistinguishable(
    db_session: Session,
    monkeypatch,
):
    company_a = _add_company(db_session)
    company_b = _add_company(db_session)
    cross_tenant = _add_scope(db_session, company_b, "scope:private")

    with pytest.raises(InvalidSearchScopeError) as unknown_exc_info:
        _resolve(
            db_session,
            monkeypatch,
            company_id=company_a.id,
            requested_scope_keys=("scope:unknown",),
        )
    with pytest.raises(InvalidSearchScopeError) as cross_tenant_exc_info:
        _resolve(
            db_session,
            monkeypatch,
            company_id=company_a.id,
            requested_scope_keys=(cross_tenant.scope_key,),
        )

    assert type(unknown_exc_info.value) is InvalidSearchScopeError
    assert type(cross_tenant_exc_info.value) is InvalidSearchScopeError
    assert str(unknown_exc_info.value) == str(cross_tenant_exc_info.value)
    assert str(unknown_exc_info.value) == "Invalid search scope selection"


def test_mixed_valid_and_unknown_scopes_fail_as_a_whole(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    valid = _add_scope(db_session, company, "scope:valid")

    with pytest.raises(InvalidSearchScopeError):
        _resolve(
            db_session,
            monkeypatch,
            company_id=company.id,
            requested_scope_keys=(valid.scope_key, "scope:unknown"),
        )


def test_mixed_valid_and_cross_tenant_scopes_fail_as_a_whole(
    db_session: Session,
    monkeypatch,
):
    company_a = _add_company(db_session)
    company_b = _add_company(db_session)
    valid = _add_scope(db_session, company_a, "scope:valid")
    cross_tenant = _add_scope(db_session, company_b, "scope:private")

    with pytest.raises(InvalidSearchScopeError):
        _resolve(
            db_session,
            monkeypatch,
            company_id=company_a.id,
            requested_scope_keys=(valid.scope_key, cross_tenant.scope_key),
        )


def test_unexpected_session_failure_is_translated(monkeypatch):
    def fail_session_creation():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        scope_resolution,
        "SessionLocal",
        fail_session_creation,
    )

    with pytest.raises(SearchScopeResolutionFailedError) as exc_info:
        scope_resolution.resolve_search_execution_context(
            company_id=1,
            requested_scope_keys=None,
        )

    assert str(exc_info.value) == "Search scope resolution failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_unexpected_query_failure_is_translated_without_details(monkeypatch):
    class QueryFailingSession:
        def scalars(self, statement):
            raise RuntimeError("database details")

    monkeypatch.setattr(
        scope_resolution,
        "SessionLocal",
        lambda: _ExistingSessionContext(QueryFailingSession()),
    )

    with pytest.raises(SearchScopeResolutionFailedError) as exc_info:
        scope_resolution.resolve_search_execution_context(
            company_id=1,
            requested_scope_keys=None,
        )

    assert str(exc_info.value) == "Search scope resolution failed"
    assert "database details" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_resolution_does_not_require_inventory_presence_rows(
    db_session: Session,
    monkeypatch,
):
    company = _add_company(db_session)
    scope = _add_scope(db_session, company, "scope:empty-inventory")

    context = _resolve(
        db_session,
        monkeypatch,
        company_id=company.id,
        requested_scope_keys=(scope.scope_key,),
    )

    assert context.resolved_scope_keys == (scope.scope_key,)


class _ExistingSessionContext:
    def __init__(self, session: Session):
        self.session = session

    def __enter__(self) -> Session:
        return self.session

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


def _resolve(
    session: Session,
    monkeypatch,
    *,
    company_id: int,
    requested_scope_keys: tuple[str, ...] | None,
):
    monkeypatch.setattr(
        scope_resolution,
        "SessionLocal",
        lambda: _ExistingSessionContext(session),
    )
    return scope_resolution.resolve_search_execution_context(
        company_id=company_id,
        requested_scope_keys=requested_scope_keys,
    )


def _add_company(session: Session) -> Company:
    token = uuid4().hex
    company = Company(id=_test_id(), name=f"Company {token}")
    session.add(company)
    session.flush()
    return company


def _add_scope(
    session: Session,
    company: Company,
    scope_key: str,
) -> InventoryScope:
    inventory_scope = InventoryScope(
        id=_test_id(),
        company_id=company.id,
        compound_id=None,
        scope_key=scope_key,
        source_identifier=f"source:{scope_key}",
    )
    session.add(inventory_scope)
    session.flush()
    return inventory_scope


def _test_id() -> int:
    return -((uuid4().int % 1_000_000_000) + 1)
