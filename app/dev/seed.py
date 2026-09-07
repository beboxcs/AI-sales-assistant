"""Deterministic development search-acceptance seed.

Run explicitly with ``python -m app.dev.seed``.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.models import (
    Company,
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitInventoryState,
)

DEVELOPMENT_COMPANY_NAME = "Development Search Acceptance V1"
SEED_SOURCE_LAST_UPDATED = date(2026, 1, 1)
SEED_LAST_SEEN_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


class DevelopmentSeedConflictError(RuntimeError):
    """Raised when the seed's stable identity conflicts with unrelated data."""


@dataclass(frozen=True)
class SeedCounts:
    companies: int
    scopes: int
    compounds: int
    units: int
    payment_plans: int
    unit_inventory_states: int
    payment_plan_inventory_states: int


@dataclass(frozen=True)
class _ScopeSeed:
    scope_key: str
    source_identifier: str
    compound_key: str


@dataclass(frozen=True)
class _CompoundSeed:
    compound_key: str
    name: str
    location: str


@dataclass(frozen=True)
class _PaymentPlanSeed:
    plan_key: str
    payment_method: PaymentMethod
    total_price: Decimal
    down_payment: Decimal | None
    installment_years: Decimal | None


@dataclass(frozen=True)
class _UnitSeed:
    external_unit_id: str
    compound_key: str
    scope_key: str
    property_type: str
    bedrooms: int
    bathrooms: int
    area_sqm: Decimal
    base_price: Decimal
    status: UnitStatus
    is_current: bool
    payment_plans: tuple[_PaymentPlanSeed, ...] = ()


_COMPOUNDS = (
    _CompoundSeed("palm-east", "Palm East", "New Cairo"),
    _CompoundSeed("zayed-heights", "Zayed Heights", "Sheikh Zayed"),
)

_SCOPES = (
    _ScopeSeed(
        "scope-nc-main",
        "development-search-acceptance-v1:scope-nc-main",
        "palm-east",
    ),
    _ScopeSeed(
        "scope-zayed-main",
        "development-search-acceptance-v1:scope-zayed-main",
        "zayed-heights",
    ),
)

_UNITS = (
    _UnitSeed(
        "A101",
        "palm-east",
        "scope-nc-main",
        "Villa",
        4,
        3,
        Decimal("250"),
        Decimal("12000000"),
        UnitStatus.AVAILABLE,
        True,
        (
            _PaymentPlanSeed(
                "A101-cash",
                PaymentMethod.CASH,
                Decimal("11800000"),
                None,
                None,
            ),
            _PaymentPlanSeed(
                "A101-inst-5y",
                PaymentMethod.INSTALLMENT,
                Decimal("13000000"),
                Decimal("1500000"),
                Decimal("5"),
            ),
        ),
    ),
    _UnitSeed(
        "A102",
        "palm-east",
        "scope-nc-main",
        "Villa",
        4,
        3,
        Decimal("260"),
        Decimal("12500000"),
        UnitStatus.AVAILABLE,
        True,
        (
            _PaymentPlanSeed(
                "A102-cash",
                PaymentMethod.CASH,
                Decimal("12200000"),
                None,
                None,
            ),
            _PaymentPlanSeed(
                "A102-inst-6y",
                PaymentMethod.INSTALLMENT,
                Decimal("13800000"),
                Decimal("2000000"),
                Decimal("6"),
            ),
        ),
    ),
    _UnitSeed(
        "A103",
        "palm-east",
        "scope-nc-main",
        "Villa",
        3,
        3,
        Decimal("240"),
        Decimal("11500000"),
        UnitStatus.AVAILABLE,
        True,
        (
            _PaymentPlanSeed(
                "A103-inst-5y",
                PaymentMethod.INSTALLMENT,
                Decimal("12400000"),
                Decimal("1200000"),
                Decimal("5"),
            ),
        ),
    ),
    _UnitSeed(
        "A104",
        "palm-east",
        "scope-nc-main",
        "Apartment",
        4,
        3,
        Decimal("220"),
        Decimal("10000000"),
        UnitStatus.AVAILABLE,
        True,
        (
            _PaymentPlanSeed(
                "A104-inst-7y",
                PaymentMethod.INSTALLMENT,
                Decimal("11000000"),
                Decimal("900000"),
                Decimal("7"),
            ),
        ),
    ),
    _UnitSeed(
        "A105",
        "palm-east",
        "scope-nc-main",
        "Villa",
        4,
        3,
        Decimal("255"),
        Decimal("10500000"),
        UnitStatus.HOLD,
        True,
    ),
    _UnitSeed(
        "A106",
        "palm-east",
        "scope-nc-main",
        "Villa",
        4,
        3,
        Decimal("255"),
        Decimal("10800000"),
        UnitStatus.AVAILABLE,
        False,
    ),
    _UnitSeed(
        "Z101",
        "zayed-heights",
        "scope-zayed-main",
        "Villa",
        4,
        3,
        Decimal("250"),
        Decimal("11000000"),
        UnitStatus.AVAILABLE,
        True,
    ),
)

_EXPECTED_SCOPE_KEYS = frozenset(spec.scope_key for spec in _SCOPES)
_EXPECTED_COMPOUND_KEYS = frozenset(spec.compound_key for spec in _COMPOUNDS)
_EXPECTED_UNIT_IDENTITIES = frozenset(
    (spec.compound_key, spec.external_unit_id) for spec in _UNITS
)
_EXPECTED_PLAN_IDENTITIES = frozenset(
    (unit_spec.compound_key, unit_spec.external_unit_id, plan_spec.plan_key)
    for unit_spec in _UNITS
    for plan_spec in unit_spec.payment_plans
)
_EXPECTED_UNIT_STATE_IDENTITIES = frozenset(
    (spec.compound_key, spec.external_unit_id, spec.scope_key)
    for spec in _UNITS
)
_EXPECTED_PLAN_STATE_IDENTITIES = frozenset(
    (
        unit_spec.compound_key,
        unit_spec.external_unit_id,
        plan_spec.plan_key,
        unit_spec.scope_key,
    )
    for unit_spec in _UNITS
    for plan_spec in unit_spec.payment_plans
)


def seed_development_search_acceptance_data(session: Session) -> SeedCounts:
    """Reconcile seed-owned records without committing the caller's transaction."""
    company = _resolve_company(session)
    _assert_no_unexpected_seed_tenant_rows(session, company.id)

    compounds = {
        spec.compound_key: _upsert_compound(session, company, spec)
        for spec in _COMPOUNDS
    }
    scopes = {
        spec.scope_key: _upsert_scope(
            session,
            company,
            compounds[spec.compound_key],
            spec,
        )
        for spec in _SCOPES
    }

    units: list[Unit] = []
    payment_plans: list[PaymentPlan] = []
    unit_states: list[UnitInventoryState] = []
    plan_states: list[PaymentPlanInventoryState] = []
    for spec in _UNITS:
        unit = _upsert_unit(session, company, compounds[spec.compound_key], spec)
        units.append(unit)
        scope = scopes[spec.scope_key]
        unit_states.append(
            _upsert_unit_state(session, company, unit, scope, spec.is_current)
        )

        for plan_spec in spec.payment_plans:
            payment_plan = _upsert_payment_plan(
                session,
                company,
                unit,
                plan_spec,
            )
            payment_plans.append(payment_plan)
            plan_states.append(
                _upsert_plan_state(
                    session,
                    company,
                    payment_plan,
                    scope,
                    spec.is_current,
                )
            )

    session.flush()
    return SeedCounts(
        companies=1,
        scopes=len(scopes),
        compounds=len(compounds),
        units=len(units),
        payment_plans=len(payment_plans),
        unit_inventory_states=len(unit_states),
        payment_plan_inventory_states=len(plan_states),
    )


def run_seed() -> SeedCounts:
    """Apply the seed atomically using the application's configured database."""
    with SessionLocal.begin() as session:
        return seed_development_search_acceptance_data(session)


def main() -> None:
    counts = run_seed()
    print("Development Search Acceptance Seed V1 applied:")
    for field_name, value in counts.__dict__.items():
        print(f"  {field_name}: {value}")


def _resolve_company(session: Session) -> Company:
    statement = select(Company).where(
        or_(
            Company.id == settings.DEV_COMPANY_ID,
            Company.name == DEVELOPMENT_COMPANY_NAME,
        )
    )
    matches = list(session.scalars(statement))
    if len(matches) > 1:
        raise DevelopmentSeedConflictError(
            "Development seed company identity is ambiguous"
        )
    if matches:
        company = matches[0]
        if (
            company.id != settings.DEV_COMPANY_ID
            or company.name != DEVELOPMENT_COMPANY_NAME
        ):
            raise DevelopmentSeedConflictError(
                "Development seed company identity conflicts with existing data"
            )
        return company

    company = Company(
        id=settings.DEV_COMPANY_ID,
        name=DEVELOPMENT_COMPANY_NAME,
    )
    session.add(company)
    session.flush()
    return company


def _assert_no_unexpected_seed_tenant_rows(
    session: Session,
    company_id: int,
) -> None:
    _reject_unexpected(
        "inventory scope",
        set(
            session.scalars(
                select(InventoryScope.scope_key).where(
                    InventoryScope.company_id == company_id
                )
            )
        ),
        _EXPECTED_SCOPE_KEYS,
    )
    _reject_unexpected(
        "compound",
        set(
            session.scalars(
                select(Compound.compound_key).where(
                    Compound.company_id == company_id
                )
            )
        ),
        _EXPECTED_COMPOUND_KEYS,
    )
    _reject_unexpected(
        "unit",
        {
            tuple(row)
            for row in session.execute(
                select(Compound.compound_key, Unit.external_unit_id)
                .select_from(Unit)
                .join(
                    Compound,
                    and_(
                        Compound.company_id == Unit.company_id,
                        Compound.id == Unit.compound_id,
                    ),
                )
                .where(Unit.company_id == company_id)
            )
        },
        _EXPECTED_UNIT_IDENTITIES,
    )
    _reject_unexpected(
        "payment plan",
        {
            tuple(row)
            for row in session.execute(
                select(
                    Compound.compound_key,
                    Unit.external_unit_id,
                    PaymentPlan.plan_key,
                )
                .select_from(PaymentPlan)
                .join(
                    Unit,
                    and_(
                        Unit.company_id == PaymentPlan.company_id,
                        Unit.id == PaymentPlan.unit_id,
                    ),
                )
                .join(
                    Compound,
                    and_(
                        Compound.company_id == Unit.company_id,
                        Compound.id == Unit.compound_id,
                    ),
                )
                .where(PaymentPlan.company_id == company_id)
            )
        },
        _EXPECTED_PLAN_IDENTITIES,
    )
    _reject_unexpected(
        "unit inventory state",
        {
            tuple(row)
            for row in session.execute(
                select(
                    Compound.compound_key,
                    Unit.external_unit_id,
                    InventoryScope.scope_key,
                )
                .select_from(UnitInventoryState)
                .join(
                    Unit,
                    and_(
                        Unit.company_id == UnitInventoryState.company_id,
                        Unit.id == UnitInventoryState.unit_id,
                    ),
                )
                .join(
                    Compound,
                    and_(
                        Compound.company_id == Unit.company_id,
                        Compound.id == Unit.compound_id,
                    ),
                )
                .join(
                    InventoryScope,
                    and_(
                        InventoryScope.company_id
                        == UnitInventoryState.company_id,
                        InventoryScope.id
                        == UnitInventoryState.inventory_scope_id,
                    ),
                )
                .where(UnitInventoryState.company_id == company_id)
            )
        },
        _EXPECTED_UNIT_STATE_IDENTITIES,
    )
    _reject_unexpected(
        "payment-plan inventory state",
        {
            tuple(row)
            for row in session.execute(
                select(
                    Compound.compound_key,
                    Unit.external_unit_id,
                    PaymentPlan.plan_key,
                    InventoryScope.scope_key,
                )
                .select_from(PaymentPlanInventoryState)
                .join(
                    PaymentPlan,
                    and_(
                        PaymentPlan.company_id
                        == PaymentPlanInventoryState.company_id,
                        PaymentPlan.id
                        == PaymentPlanInventoryState.payment_plan_id,
                    ),
                )
                .join(
                    Unit,
                    and_(
                        Unit.company_id == PaymentPlan.company_id,
                        Unit.id == PaymentPlan.unit_id,
                    ),
                )
                .join(
                    Compound,
                    and_(
                        Compound.company_id == Unit.company_id,
                        Compound.id == Unit.compound_id,
                    ),
                )
                .join(
                    InventoryScope,
                    and_(
                        InventoryScope.company_id
                        == PaymentPlanInventoryState.company_id,
                        InventoryScope.id
                        == PaymentPlanInventoryState.inventory_scope_id,
                    ),
                )
                .where(PaymentPlanInventoryState.company_id == company_id)
            )
        },
        _EXPECTED_PLAN_STATE_IDENTITIES,
    )


def _reject_unexpected(
    record_type: str,
    actual_identities: set,
    expected_identities: frozenset,
) -> None:
    if actual_identities - expected_identities:
        raise DevelopmentSeedConflictError(
            f"Unexpected {record_type} rows exist for development seed tenant"
        )


def _upsert_compound(
    session: Session,
    company: Company,
    spec: _CompoundSeed,
) -> Compound:
    compound = session.scalar(
        select(Compound).where(
            Compound.company_id == company.id,
            Compound.compound_key == spec.compound_key,
        )
    )
    if compound is None:
        compound = Compound(
            company_id=company.id,
            compound_key=spec.compound_key,
        )
        session.add(compound)

    compound.name = spec.name
    compound.location = spec.location
    if compound.id is None:
        session.flush()
    return compound


def _upsert_scope(
    session: Session,
    company: Company,
    compound: Compound,
    spec: _ScopeSeed,
) -> InventoryScope:
    scope = session.scalar(
        select(InventoryScope).where(
            InventoryScope.company_id == company.id,
            InventoryScope.scope_key == spec.scope_key,
        )
    )
    if scope is None:
        scope = InventoryScope(
            company_id=company.id,
            scope_key=spec.scope_key,
        )
        session.add(scope)

    scope.compound_id = compound.id
    scope.source_identifier = spec.source_identifier
    if scope.id is None:
        session.flush()
    return scope


def _upsert_unit(
    session: Session,
    company: Company,
    compound: Compound,
    spec: _UnitSeed,
) -> Unit:
    unit = session.scalar(
        select(Unit).where(
            Unit.company_id == company.id,
            Unit.compound_id == compound.id,
            Unit.external_unit_id == spec.external_unit_id,
        )
    )
    if unit is None:
        unit = Unit(
            company_id=company.id,
            compound_id=compound.id,
            external_unit_id=spec.external_unit_id,
        )
        session.add(unit)

    unit.property_type = spec.property_type
    unit.bedrooms = spec.bedrooms
    unit.bathrooms = spec.bathrooms
    unit.area_sqm = spec.area_sqm
    unit.base_price = spec.base_price
    unit.base_currency = Currency.EGP.value
    unit.status = spec.status.value
    unit.delivery_date = None
    unit.source_last_updated = SEED_SOURCE_LAST_UPDATED
    if unit.id is None:
        session.flush()
    return unit


def _upsert_payment_plan(
    session: Session,
    company: Company,
    unit: Unit,
    spec: _PaymentPlanSeed,
) -> PaymentPlan:
    payment_plan = session.scalar(
        select(PaymentPlan).where(
            PaymentPlan.company_id == company.id,
            PaymentPlan.unit_id == unit.id,
            PaymentPlan.plan_key == spec.plan_key,
        )
    )
    if payment_plan is None:
        payment_plan = PaymentPlan(
            company_id=company.id,
            unit_id=unit.id,
            plan_key=spec.plan_key,
        )
        session.add(payment_plan)

    payment_plan.source_plan_id = None
    payment_plan.plan_name = None
    payment_plan.payment_method = spec.payment_method.value
    payment_plan.total_price = spec.total_price
    payment_plan.down_payment = spec.down_payment
    payment_plan.installment_years = spec.installment_years
    payment_plan.currency = Currency.EGP.value
    if payment_plan.id is None:
        session.flush()
    return payment_plan


def _upsert_unit_state(
    session: Session,
    company: Company,
    unit: Unit,
    scope: InventoryScope,
    is_current: bool,
) -> UnitInventoryState:
    state = session.scalar(
        select(UnitInventoryState).where(
            UnitInventoryState.company_id == company.id,
            UnitInventoryState.unit_id == unit.id,
            UnitInventoryState.inventory_scope_id == scope.id,
        )
    )
    if state is None:
        state = UnitInventoryState(
            company_id=company.id,
            unit_id=unit.id,
            inventory_scope_id=scope.id,
        )
        session.add(state)

    state.is_present_in_latest_inventory = is_current
    state.last_seen_at = SEED_LAST_SEEN_AT
    if state.id is None:
        session.flush()
    return state


def _upsert_plan_state(
    session: Session,
    company: Company,
    payment_plan: PaymentPlan,
    scope: InventoryScope,
    is_current: bool,
) -> PaymentPlanInventoryState:
    state = session.scalar(
        select(PaymentPlanInventoryState).where(
            PaymentPlanInventoryState.company_id == company.id,
            PaymentPlanInventoryState.payment_plan_id == payment_plan.id,
            PaymentPlanInventoryState.inventory_scope_id == scope.id,
        )
    )
    if state is None:
        state = PaymentPlanInventoryState(
            company_id=company.id,
            payment_plan_id=payment_plan.id,
            inventory_scope_id=scope.id,
        )
        session.add(state)

    state.is_present_in_latest_inventory = is_current
    state.last_seen_at = SEED_LAST_SEEN_AT
    if state.id is None:
        session.flush()
    return state


if __name__ == "__main__":
    main()
