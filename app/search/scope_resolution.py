"""Tenant-safe resolution of backend search execution scopes."""

from sqlalchemy import select

from app.database import SessionLocal
from app.models import InventoryScope
from app.search.contracts import SearchExecutionContext
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchScopeResolutionFailedError,
)


def resolve_search_execution_context(
    *,
    company_id: int,
    requested_scope_keys: tuple[str, ...] | None,
) -> SearchExecutionContext:
    """Resolve a tenant-owned scope selection into an execution context."""
    if requested_scope_keys == ():
        return SearchExecutionContext(company_id=company_id, resolved_scope_keys=())

    if requested_scope_keys is not None:
        _validate_explicit_scope_keys(requested_scope_keys)

    try:
        with SessionLocal() as session:
            if requested_scope_keys is None:
                statement = (
                    select(InventoryScope.scope_key)
                    .where(InventoryScope.company_id == company_id)
                    .order_by(InventoryScope.id.asc())
                )
                resolved_scope_keys = tuple(session.scalars(statement).all())
            else:
                statement = select(InventoryScope.scope_key).where(
                    InventoryScope.company_id == company_id,
                    InventoryScope.scope_key.in_(requested_scope_keys),
                )
                resolved_keys = set(session.scalars(statement).all())
                if len(resolved_keys) != len(requested_scope_keys):
                    raise InvalidSearchScopeError(
                        "Invalid search scope selection"
                    )
                resolved_scope_keys = tuple(
                    scope_key
                    for scope_key in requested_scope_keys
                    if scope_key in resolved_keys
                )

        return SearchExecutionContext(
            company_id=company_id,
            resolved_scope_keys=resolved_scope_keys,
        )
    except InvalidSearchScopeError:
        raise
    except Exception as exc:
        raise SearchScopeResolutionFailedError(
            "Search scope resolution failed"
        ) from exc


def _validate_explicit_scope_keys(requested_scope_keys: tuple[str, ...]) -> None:
    if any(
        not isinstance(scope_key, str) or not scope_key.strip()
        for scope_key in requested_scope_keys
    ):
        raise InvalidSearchScopeError("Invalid search scope selection")
    if len(set(requested_scope_keys)) != len(requested_scope_keys):
        raise InvalidSearchScopeError("Invalid search scope selection")
