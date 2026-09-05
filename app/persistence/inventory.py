"""Atomic persistence for normalized inventory snapshots."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai_engine.change_detector import InventoryChange
from app.ai_engine.schemas import InventorySnapshot, PaymentPlanSchema, UnitSchema
from app.models import (
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitHistory,
    UnitInventoryState,
)


class InventoryPersistenceError(Exception):
    """Base class for inventory persistence failures."""


class InventoryScopeNotFoundError(InventoryPersistenceError, LookupError):
    """Raised when an inventory scope is absent from the requested company."""


class CompoundNotFoundError(InventoryPersistenceError, LookupError):
    """Raised when a canonical compound key cannot be resolved for a company."""


class CompoundDescriptionMismatchError(InventoryPersistenceError, ValueError):
    """Raised when incoming compound descriptions conflict with persisted data."""


class InventoryUnitNotFoundError(InventoryPersistenceError, LookupError):
    """Raised when a change references a unit outside its canonical identity."""


class InventoryPersistenceFailedError(InventoryPersistenceError):
    """Raised when an unexpected error prevents atomic snapshot persistence."""


def get_inventory_scope(
    session: Session,
    *,
    company_id: int,
    scope_key: str,
) -> InventoryScope:
    """Return an existing inventory scope belonging to the requested company."""
    statement = select(InventoryScope).where(
        InventoryScope.company_id == company_id,
        InventoryScope.scope_key == scope_key,
    )
    inventory_scope = session.scalar(statement)

    if inventory_scope is None:
        raise InventoryScopeNotFoundError(
            f"Inventory scope {scope_key!r} was not found for company {company_id}"
        )

    return inventory_scope


def get_compound(
    session: Session,
    *,
    company_id: int,
    compound_key: str,
) -> Compound:
    """Resolve a compound strictly by its tenant-scoped canonical key."""
    statement = select(Compound).where(
        Compound.company_id == company_id,
        Compound.compound_key == compound_key,
    )
    compound = session.scalar(statement)

    if compound is None:
        raise CompoundNotFoundError(
            f"Compound {compound_key!r} was not found for company {company_id}"
        )

    return compound


def validate_compound_description(
    compound: Compound,
    *,
    compound_name: str,
    location: str,
) -> None:
    """Reject descriptive inventory data that conflicts with the canonical row."""
    conflicts: list[str] = []
    if compound.name != compound_name:
        conflicts.append(
            f"compound_name {compound_name!r} does not match {compound.name!r}"
        )
    if compound.location != location:
        conflicts.append(
            f"location {location!r} does not match {compound.location!r}"
        )

    if conflicts:
        raise CompoundDescriptionMismatchError(
            f"Compound {compound.compound_key!r} descriptive data conflicts: "
            + "; ".join(conflicts)
        )


def get_unit_by_identity(
    session: Session,
    *,
    company_id: int,
    compound_id: int,
    external_unit_id: str,
) -> Unit | None:
    """Look up a unit by its complete persisted tenant identity."""
    statement = select(Unit).where(
        Unit.company_id == company_id,
        Unit.compound_id == compound_id,
        Unit.external_unit_id == external_unit_id,
    )
    return session.scalar(statement)


def upsert_unit(
    session: Session,
    *,
    company_id: int,
    compound: Compound,
    unit_data: UnitSchema,
) -> Unit:
    """Insert or update the current fields of one normalized unit."""
    unit = get_unit_by_identity(
        session,
        company_id=company_id,
        compound_id=compound.id,
        external_unit_id=unit_data.external_unit_id,
    )
    if unit is None:
        unit = Unit(
            company_id=company_id,
            compound_id=compound.id,
            external_unit_id=unit_data.external_unit_id,
        )
        session.add(unit)

    unit.property_type = unit_data.property_type
    unit.base_price = unit_data.base_price
    unit.base_currency = unit_data.base_currency.value
    unit.status = unit_data.status.value
    unit.bedrooms = unit_data.bedrooms
    unit.bathrooms = unit_data.bathrooms
    unit.area_sqm = unit_data.area_sqm
    unit.delivery_date = unit_data.delivery_date
    unit.source_last_updated = unit_data.last_updated

    if unit.id is None:
        session.flush()

    return unit


def upsert_payment_plan(
    session: Session,
    *,
    company_id: int,
    unit: Unit,
    plan_data: PaymentPlanSchema,
) -> PaymentPlan:
    """Insert or update a payment plan by unit and canonical plan key."""
    statement = select(PaymentPlan).where(
        PaymentPlan.company_id == company_id,
        PaymentPlan.unit_id == unit.id,
        PaymentPlan.plan_key == plan_data.plan_key,
    )
    payment_plan = session.scalar(statement)
    if payment_plan is None:
        payment_plan = PaymentPlan(
            company_id=company_id,
            unit_id=unit.id,
            plan_key=plan_data.plan_key,
        )
        session.add(payment_plan)

    payment_plan.source_plan_id = plan_data.source_plan_id
    payment_plan.plan_name = plan_data.plan_name
    payment_plan.payment_method = plan_data.payment_method.value
    payment_plan.down_payment = plan_data.down_payment
    payment_plan.installment_years = plan_data.installment_years
    payment_plan.total_price = plan_data.total_price
    payment_plan.currency = plan_data.currency.value

    if payment_plan.id is None:
        session.flush()

    return payment_plan


def persist_inventory_snapshot(
    session: Session,
    *,
    company_id: int,
    snapshot: InventorySnapshot,
    changes: list[InventoryChange],
) -> None:
    """Persist one authoritative inventory snapshot as one atomic transaction."""
    operation_timestamp = datetime.now(timezone.utc)

    try:
        inventory_scope = get_inventory_scope(
            session,
            company_id=company_id,
            scope_key=snapshot.scope_key,
        )
        observed_unit_ids: set[int] = set()
        compounds_by_key: dict[str, Compound] = {}

        for unit_data in snapshot.units:
            compound = compounds_by_key.get(unit_data.compound_key)
            if compound is None:
                compound = get_compound(
                    session,
                    company_id=company_id,
                    compound_key=unit_data.compound_key,
                )
                compounds_by_key[unit_data.compound_key] = compound

            validate_compound_description(
                compound,
                compound_name=unit_data.compound_name,
                location=unit_data.location,
            )
            unit = upsert_unit(
                session,
                company_id=company_id,
                compound=compound,
                unit_data=unit_data,
            )
            observed_unit_ids.add(unit.id)
            _mark_unit_observed(
                session,
                company_id=company_id,
                unit_id=unit.id,
                inventory_scope_id=inventory_scope.id,
                operation_timestamp=operation_timestamp,
            )

            observed_plan_ids: set[int] = set()
            for plan_data in unit_data.payment_plans:
                payment_plan = upsert_payment_plan(
                    session,
                    company_id=company_id,
                    unit=unit,
                    plan_data=plan_data,
                )
                observed_plan_ids.add(payment_plan.id)
                _mark_payment_plan_observed(
                    session,
                    company_id=company_id,
                    payment_plan_id=payment_plan.id,
                    inventory_scope_id=inventory_scope.id,
                    operation_timestamp=operation_timestamp,
                )

            _mark_unobserved_plans_missing(
                session,
                company_id=company_id,
                unit_id=unit.id,
                inventory_scope_id=inventory_scope.id,
                observed_plan_ids=observed_plan_ids,
            )

        _mark_unobserved_units_missing(
            session,
            company_id=company_id,
            inventory_scope_id=inventory_scope.id,
            observed_unit_ids=observed_unit_ids,
        )
        _persist_history(
            session,
            company_id=company_id,
            inventory_scope_id=inventory_scope.id,
            changes=changes,
            operation_timestamp=operation_timestamp,
            compounds_by_key=compounds_by_key,
        )
        session.commit()
    except InventoryPersistenceError:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        raise InventoryPersistenceFailedError(
            "Inventory snapshot persistence failed"
        ) from exc


def _mark_unit_observed(
    session: Session,
    *,
    company_id: int,
    unit_id: int,
    inventory_scope_id: int,
    operation_timestamp: datetime,
) -> None:
    statement = select(UnitInventoryState).where(
        UnitInventoryState.company_id == company_id,
        UnitInventoryState.unit_id == unit_id,
        UnitInventoryState.inventory_scope_id == inventory_scope_id,
    )
    state = session.scalar(statement)
    if state is None:
        state = UnitInventoryState(
            company_id=company_id,
            unit_id=unit_id,
            inventory_scope_id=inventory_scope_id,
        )
        session.add(state)

    state.is_present_in_latest_inventory = True
    state.last_seen_at = operation_timestamp


def _mark_payment_plan_observed(
    session: Session,
    *,
    company_id: int,
    payment_plan_id: int,
    inventory_scope_id: int,
    operation_timestamp: datetime,
) -> None:
    statement = select(PaymentPlanInventoryState).where(
        PaymentPlanInventoryState.company_id == company_id,
        PaymentPlanInventoryState.payment_plan_id == payment_plan_id,
        PaymentPlanInventoryState.inventory_scope_id == inventory_scope_id,
    )
    state = session.scalar(statement)
    if state is None:
        state = PaymentPlanInventoryState(
            company_id=company_id,
            payment_plan_id=payment_plan_id,
            inventory_scope_id=inventory_scope_id,
        )
        session.add(state)

    state.is_present_in_latest_inventory = True
    state.last_seen_at = operation_timestamp


def _mark_unobserved_plans_missing(
    session: Session,
    *,
    company_id: int,
    unit_id: int,
    inventory_scope_id: int,
    observed_plan_ids: set[int],
) -> None:
    statement = (
        select(PaymentPlanInventoryState)
        .join(
            PaymentPlan,
            PaymentPlan.id == PaymentPlanInventoryState.payment_plan_id,
        )
        .where(
            PaymentPlanInventoryState.company_id == company_id,
            PaymentPlanInventoryState.inventory_scope_id == inventory_scope_id,
            PaymentPlan.company_id == company_id,
            PaymentPlan.unit_id == unit_id,
        )
    )
    for state in session.scalars(statement):
        if state.payment_plan_id not in observed_plan_ids:
            state.is_present_in_latest_inventory = False


def _mark_unobserved_units_missing(
    session: Session,
    *,
    company_id: int,
    inventory_scope_id: int,
    observed_unit_ids: set[int],
) -> None:
    statement = select(UnitInventoryState).where(
        UnitInventoryState.company_id == company_id,
        UnitInventoryState.inventory_scope_id == inventory_scope_id,
    )
    for state in session.scalars(statement):
        if state.unit_id in observed_unit_ids:
            continue
        state.is_present_in_latest_inventory = False
        _mark_all_unit_plans_missing(
            session,
            company_id=company_id,
            unit_id=state.unit_id,
            inventory_scope_id=inventory_scope_id,
        )


def _mark_all_unit_plans_missing(
    session: Session,
    *,
    company_id: int,
    unit_id: int,
    inventory_scope_id: int,
) -> None:
    statement = (
        select(PaymentPlanInventoryState)
        .join(
            PaymentPlan,
            PaymentPlan.id == PaymentPlanInventoryState.payment_plan_id,
        )
        .where(
            PaymentPlanInventoryState.company_id == company_id,
            PaymentPlanInventoryState.inventory_scope_id == inventory_scope_id,
            PaymentPlan.company_id == company_id,
            PaymentPlan.unit_id == unit_id,
        )
    )
    for state in session.scalars(statement):
        state.is_present_in_latest_inventory = False


def _persist_history(
    session: Session,
    *,
    company_id: int,
    inventory_scope_id: int,
    changes: list[InventoryChange],
    operation_timestamp: datetime,
    compounds_by_key: dict[str, Compound],
) -> None:
    for change in changes:
        compound = compounds_by_key.get(change.compound_key)
        if compound is None:
            compound = get_compound(
                session,
                company_id=company_id,
                compound_key=change.compound_key,
            )
            compounds_by_key[change.compound_key] = compound

        unit = get_unit_by_identity(
            session,
            company_id=company_id,
            compound_id=compound.id,
            external_unit_id=change.external_unit_id,
        )
        if unit is None:
            raise InventoryUnitNotFoundError(
                "Inventory change references unknown unit "
                f"{change.compound_key!r}/{change.external_unit_id!r} "
                f"for company {company_id}"
            )

        serialized_change = change.model_dump(mode="json")
        session.add(
            UnitHistory(
                company_id=company_id,
                unit_id=unit.id,
                inventory_scope_id=inventory_scope_id,
                change_type=serialized_change["change_type"],
                changes=serialized_change["changes"],
                payment_plan_changes=serialized_change["payment_plan_changes"],
                detected_at=operation_timestamp,
            )
        )
