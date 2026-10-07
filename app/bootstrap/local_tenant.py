"""Local-only atomic bootstrap for a tenant, its first admin, and catalog."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.human.passwords import hash_password, verify_password
from app.config import settings
from app.database import SessionLocal
from app.domain.enums import MembershipRole
from app.domain.users import normalize_email
from app.models import Company, CompanyMembership, Compound, InventoryScope, User


class LocalTenantBootstrapError(Exception):
    """Base error for the local tenant bootstrap boundary."""


class LocalTenantBootstrapDisabledError(LocalTenantBootstrapError):
    """Raised when bootstrap is called outside local or test environments."""


class LocalTenantBootstrapValidationError(LocalTenantBootstrapError):
    """Raised when bootstrap input is invalid."""


class LocalTenantBootstrapConflictError(LocalTenantBootstrapError):
    """Raised when persisted state conflicts with the requested bootstrap."""

    def __init__(self, resource: str):
        self.resource = resource
        super().__init__(
            f"Local tenant bootstrap conflicts with existing {resource} state"
        )


class LocalTenantBootstrapInfrastructureError(LocalTenantBootstrapError):
    """Raised when bootstrap cannot complete because its backend is unavailable."""


class BootstrapDisposition(str, Enum):
    CREATED = "created"
    REUSED = "reused"


@dataclass(frozen=True)
class CompoundBootstrapSpec:
    compound_key: str
    name: str
    location: str


@dataclass(frozen=True)
class InventoryScopeBootstrapSpec:
    scope_key: str
    source_identifier: str
    compound_key: str | None = None


@dataclass(frozen=True)
class CompoundBootstrapResult:
    compound_id: int
    compound_key: str
    disposition: BootstrapDisposition


@dataclass(frozen=True)
class InventoryScopeBootstrapResult:
    scope_id: int
    scope_key: str
    compound_id: int | None
    disposition: BootstrapDisposition


@dataclass(frozen=True)
class LocalTenantBootstrapResult:
    company_id: int
    company_disposition: BootstrapDisposition
    admin_user_id: int
    admin_user_disposition: BootstrapDisposition
    admin_membership_id: int
    admin_membership_disposition: BootstrapDisposition
    normalized_admin_email: str
    compounds: tuple[CompoundBootstrapResult, ...]
    scopes: tuple[InventoryScopeBootstrapResult, ...]


@dataclass(frozen=True)
class _ValidatedBootstrapRequest:
    company_id: int
    company_name: str
    normalized_admin_email: str
    admin_password: str = field(repr=False)
    compounds: tuple[CompoundBootstrapSpec, ...]
    scopes: tuple[InventoryScopeBootstrapSpec, ...]


_DISABLED_MESSAGE = "Local tenant bootstrap is disabled"
_INFRASTRUCTURE_MESSAGE = "Local tenant bootstrap unavailable"
_ALLOWED_ENVIRONMENTS = frozenset({"local", "test"})


def bootstrap_local_tenant(
    *,
    company_id: int,
    company_name: str,
    admin_email: str,
    admin_password: str,
    compounds: tuple[CompoundBootstrapSpec, ...],
    scopes: tuple[InventoryScopeBootstrapSpec, ...],
) -> LocalTenantBootstrapResult:
    """Create or exactly reuse one complete local tenant bootstrap atomically."""
    if settings.APP_ENV not in _ALLOWED_ENVIRONMENTS:
        raise LocalTenantBootstrapDisabledError(_DISABLED_MESSAGE)

    request = _validate_request(
        company_id=company_id,
        company_name=company_name,
        admin_email=admin_email,
        admin_password=admin_password,
        compounds=compounds,
        scopes=scopes,
    )

    try:
        with SessionLocal.begin() as session:
            _acquire_bootstrap_locks(session, request=request)
            company, company_disposition = _resolve_company(
                session,
                company_id=request.company_id,
                company_name=request.company_name,
            )
            user, user_disposition = _resolve_admin_user(
                session,
                normalized_email=request.normalized_admin_email,
                password=request.admin_password,
            )
            membership, membership_disposition = _resolve_initial_admin_membership(
                session,
                company_id=company.id,
                user_id=user.id,
            )
            compound_results, compounds_by_key = _resolve_compounds(
                session,
                company_id=company.id,
                specs=request.compounds,
            )
            scope_results = _resolve_scopes(
                session,
                company_id=company.id,
                specs=request.scopes,
                compounds_by_key=compounds_by_key,
            )
            session.flush()

            return LocalTenantBootstrapResult(
                company_id=company.id,
                company_disposition=company_disposition,
                admin_user_id=user.id,
                admin_user_disposition=user_disposition,
                admin_membership_id=membership.id,
                admin_membership_disposition=membership_disposition,
                normalized_admin_email=request.normalized_admin_email,
                compounds=compound_results,
                scopes=scope_results,
            )
    except LocalTenantBootstrapError:
        raise
    except IntegrityError:
        raise LocalTenantBootstrapConflictError("persistence") from None
    except Exception:
        raise LocalTenantBootstrapInfrastructureError(
            _INFRASTRUCTURE_MESSAGE
        ) from None


def _validate_request(
    *,
    company_id: int,
    company_name: str,
    admin_email: str,
    admin_password: str,
    compounds: tuple[CompoundBootstrapSpec, ...],
    scopes: tuple[InventoryScopeBootstrapSpec, ...],
) -> _ValidatedBootstrapRequest:
    if type(company_id) is not int or company_id <= 0:
        raise LocalTenantBootstrapValidationError(
            "company_id must be a positive integer"
        )
    canonical_company_name = _canonical_nonblank(company_name, "company_name")
    try:
        normalized_admin_email = normalize_email(admin_email)
    except (AttributeError, TypeError, ValueError):
        raise LocalTenantBootstrapValidationError("admin_email is invalid") from None
    if len(normalized_admin_email) > 320:
        raise LocalTenantBootstrapValidationError("admin_email is invalid")
    if not isinstance(admin_password, str) or not admin_password.strip():
        raise LocalTenantBootstrapValidationError("admin_password is invalid")
    if not isinstance(compounds, tuple) or not compounds:
        raise LocalTenantBootstrapValidationError(
            "compounds must be a non-empty tuple"
        )
    if not isinstance(scopes, tuple) or not scopes:
        raise LocalTenantBootstrapValidationError(
            "scopes must be a non-empty tuple"
        )

    canonical_compounds = tuple(_validate_compound_spec(spec) for spec in compounds)
    canonical_scopes = tuple(_validate_scope_spec(spec) for spec in scopes)
    _reject_duplicate_keys(
        (spec.compound_key for spec in canonical_compounds),
        field_name="compound_key",
    )
    _reject_duplicate_keys(
        (spec.scope_key for spec in canonical_scopes),
        field_name="scope_key",
    )
    return _ValidatedBootstrapRequest(
        company_id=company_id,
        company_name=canonical_company_name,
        normalized_admin_email=normalized_admin_email,
        admin_password=admin_password,
        compounds=tuple(
            sorted(canonical_compounds, key=lambda spec: spec.compound_key)
        ),
        scopes=tuple(sorted(canonical_scopes, key=lambda spec: spec.scope_key)),
    )


def _validate_compound_spec(spec: object) -> CompoundBootstrapSpec:
    if not isinstance(spec, CompoundBootstrapSpec):
        raise LocalTenantBootstrapValidationError("compound specification is invalid")
    return CompoundBootstrapSpec(
        compound_key=_canonical_nonblank(spec.compound_key, "compound_key"),
        name=_canonical_nonblank(spec.name, "compound name"),
        location=_canonical_nonblank(spec.location, "compound location"),
    )


def _validate_scope_spec(spec: object) -> InventoryScopeBootstrapSpec:
    if not isinstance(spec, InventoryScopeBootstrapSpec):
        raise LocalTenantBootstrapValidationError("scope specification is invalid")
    compound_key = (
        None
        if spec.compound_key is None
        else _canonical_nonblank(spec.compound_key, "scope compound_key")
    )
    return InventoryScopeBootstrapSpec(
        scope_key=_canonical_nonblank(spec.scope_key, "scope_key"),
        source_identifier=_canonical_nonblank(
            spec.source_identifier,
            "source_identifier",
        ),
        compound_key=compound_key,
    )


def _canonical_nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise LocalTenantBootstrapValidationError(f"{field_name} must be a string")
    canonical = value.strip()
    if not canonical:
        raise LocalTenantBootstrapValidationError(f"{field_name} must not be blank")
    return canonical


def _reject_duplicate_keys(keys, *, field_name: str) -> None:
    observed: set[str] = set()
    for key in keys:
        if key in observed:
            raise LocalTenantBootstrapValidationError(
                f"duplicate {field_name} in bootstrap request"
            )
        observed.add(key)


def _acquire_bootstrap_locks(
    session: Session,
    *,
    request: _ValidatedBootstrapRequest,
) -> None:
    identities = {
        f"company-id:{request.company_id}",
        f"company-name:{request.company_name}",
        f"admin-email:{request.normalized_admin_email}",
    }
    lock_keys = sorted(_advisory_lock_key(identity) for identity in identities)
    for lock_key in lock_keys:
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )


def _advisory_lock_key(identity: str) -> int:
    digest = hashlib.sha256(
        f"local-tenant-bootstrap-v1\0{identity}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _resolve_company(
    session: Session,
    *,
    company_id: int,
    company_name: str,
) -> tuple[Company, BootstrapDisposition]:
    statement = (
        select(Company)
        .where(or_(Company.id == company_id, Company.name == company_name))
        .order_by(Company.id)
        .with_for_update()
    )
    matches = list(session.scalars(statement))
    if not matches:
        company = Company(id=company_id, name=company_name)
        session.add(company)
        session.flush()
        return company, BootstrapDisposition.CREATED
    if (
        len(matches) != 1
        or matches[0].id != company_id
        or matches[0].name != company_name
    ):
        raise LocalTenantBootstrapConflictError("company")
    return matches[0], BootstrapDisposition.REUSED


def _resolve_admin_user(
    session: Session,
    *,
    normalized_email: str,
    password: str,
) -> tuple[User, BootstrapDisposition]:
    statement = (
        select(User)
        .where(User.normalized_email == normalized_email)
        .with_for_update()
    )
    user = session.scalar(statement)
    if user is None:
        user = User(
            normalized_email=normalized_email,
            password_hash=hash_password(password),
            password_version=1,
            is_active=True,
            password_changed_at=_utc_now(),
        )
        session.add(user)
        session.flush()
        return user, BootstrapDisposition.CREATED
    if (
        user.is_active is not True
        or type(user.password_version) is not int
        or user.password_version <= 0
    ):
        raise LocalTenantBootstrapConflictError("admin user")
    try:
        password_matches = verify_password(password, user.password_hash)
    except (InvalidHashError, VerificationError):
        raise LocalTenantBootstrapConflictError("admin user") from None
    if not password_matches:
        raise LocalTenantBootstrapConflictError("admin user")
    return user, BootstrapDisposition.REUSED


def _resolve_initial_admin_membership(
    session: Session,
    *,
    company_id: int,
    user_id: int,
) -> tuple[CompanyMembership, BootstrapDisposition]:
    statement = (
        select(CompanyMembership)
        .where(CompanyMembership.company_id == company_id)
        .order_by(CompanyMembership.id)
        .with_for_update()
    )
    memberships = list(session.scalars(statement))
    requested_membership = next(
        (membership for membership in memberships if membership.user_id == user_id),
        None,
    )
    if requested_membership is not None:
        if (
            requested_membership.role != MembershipRole.COMPANY_ADMIN.value
            or requested_membership.is_active is not True
            or requested_membership.deactivated_at is not None
        ):
            raise LocalTenantBootstrapConflictError("admin membership")
        return requested_membership, BootstrapDisposition.REUSED
    if memberships:
        raise LocalTenantBootstrapConflictError("initial admin membership")

    membership = CompanyMembership(
        user_id=user_id,
        company_id=company_id,
        role=MembershipRole.COMPANY_ADMIN.value,
        is_active=True,
    )
    session.add(membership)
    session.flush()
    return membership, BootstrapDisposition.CREATED


def _resolve_compounds(
    session: Session,
    *,
    company_id: int,
    specs: tuple[CompoundBootstrapSpec, ...],
) -> tuple[tuple[CompoundBootstrapResult, ...], dict[str, Compound]]:
    results: list[CompoundBootstrapResult] = []
    compounds_by_key: dict[str, Compound] = {}
    for spec in specs:
        statement = (
            select(Compound)
            .where(
                Compound.company_id == company_id,
                Compound.compound_key == spec.compound_key,
            )
            .with_for_update()
        )
        compound = session.scalar(statement)
        if compound is None:
            compound = Compound(
                company_id=company_id,
                compound_key=spec.compound_key,
                name=spec.name,
                location=spec.location,
            )
            session.add(compound)
            session.flush()
            disposition = BootstrapDisposition.CREATED
        else:
            if compound.name != spec.name or compound.location != spec.location:
                raise LocalTenantBootstrapConflictError("compound")
            disposition = BootstrapDisposition.REUSED
        compounds_by_key[spec.compound_key] = compound
        results.append(
            CompoundBootstrapResult(
                compound_id=compound.id,
                compound_key=compound.compound_key,
                disposition=disposition,
            )
        )
    return tuple(results), compounds_by_key


def _resolve_scopes(
    session: Session,
    *,
    company_id: int,
    specs: tuple[InventoryScopeBootstrapSpec, ...],
    compounds_by_key: dict[str, Compound],
) -> tuple[InventoryScopeBootstrapResult, ...]:
    results: list[InventoryScopeBootstrapResult] = []
    for spec in specs:
        compound = _resolve_scope_compound(
            session,
            company_id=company_id,
            compound_key=spec.compound_key,
            compounds_by_key=compounds_by_key,
        )
        desired_compound_id = compound.id if compound is not None else None
        statement = (
            select(InventoryScope)
            .where(
                InventoryScope.company_id == company_id,
                InventoryScope.scope_key == spec.scope_key,
            )
            .with_for_update()
        )
        scope = session.scalar(statement)
        if scope is None:
            scope = InventoryScope(
                company_id=company_id,
                compound_id=desired_compound_id,
                scope_key=spec.scope_key,
                source_identifier=spec.source_identifier,
            )
            session.add(scope)
            session.flush()
            disposition = BootstrapDisposition.CREATED
        else:
            if (
                scope.source_identifier != spec.source_identifier
                or scope.compound_id != desired_compound_id
            ):
                raise LocalTenantBootstrapConflictError("inventory scope")
            disposition = BootstrapDisposition.REUSED
        results.append(
            InventoryScopeBootstrapResult(
                scope_id=scope.id,
                scope_key=scope.scope_key,
                compound_id=scope.compound_id,
                disposition=disposition,
            )
        )
    return tuple(results)


def _resolve_scope_compound(
    session: Session,
    *,
    company_id: int,
    compound_key: str | None,
    compounds_by_key: dict[str, Compound],
) -> Compound | None:
    if compound_key is None:
        return None
    if compound_key in compounds_by_key:
        return compounds_by_key[compound_key]
    statement = (
        select(Compound)
        .where(
            Compound.company_id == company_id,
            Compound.compound_key == compound_key,
        )
        .with_for_update()
    )
    compound = session.scalar(statement)
    if compound is None:
        raise LocalTenantBootstrapConflictError("scope compound")
    compounds_by_key[compound_key] = compound
    return compound


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
