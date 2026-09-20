"""Database-backed execution of one frozen search-filter set."""

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from app.ai_engine.search_filters import SearchFilters
from app.domain.enums import Currency, PaymentMethod, UnitStatus
from app.models import (
    Compound,
    InventoryScope,
    PaymentPlan,
    PaymentPlanInventoryState,
    Unit,
    UnitInventoryState,
)
from app.search.contracts import (
    PaymentPlanCandidate,
    PriceBasis,
    SearchExecutionContext,
    UnitCandidate,
    has_payment_plan_constraints,
    resolve_price_basis,
)


def search_candidates(
    session: Session,
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
    price_basis: PriceBasis | None = None,
) -> list[UnitCandidate]:
    """Return current tenant candidates for one resolved execution context."""
    resolved_price_basis = resolve_price_basis(
        filters=filters,
        price_basis=price_basis,
    )
    if not context.resolved_scope_keys:
        return []

    unit_rows = session.execute(
        _eligible_units_statement(
            filters=filters,
            context=context,
            price_basis=resolved_price_basis,
        )
    ).all()
    if not unit_rows:
        return []

    units_by_id: dict[int, tuple[Unit, Compound]] = {}
    unit_scope_keys: dict[int, set[str]] = {}
    for unit, compound, scope_key in unit_rows:
        units_by_id[unit.id] = (unit, compound)
        unit_scope_keys.setdefault(unit.id, set()).add(scope_key)

    plan_rows = session.execute(
        _eligible_plans_statement(
            filters=filters,
            context=context,
            unit_ids=tuple(units_by_id),
            price_basis=resolved_price_basis,
        )
    ).all()
    plans_by_unit_id: dict[int, dict[int, PaymentPlan]] = {}
    plan_scope_keys: dict[int, set[str]] = {}
    for payment_plan, scope_key in plan_rows:
        plans_by_unit_id.setdefault(payment_plan.unit_id, {})[
            payment_plan.id
        ] = payment_plan
        plan_scope_keys.setdefault(payment_plan.unit_id, set()).add(scope_key)

    plan_semantics_active = (
        resolved_price_basis is PriceBasis.PLAN_TOTAL_PRICE
        and (
            has_payment_plan_constraints(filters)
            or _has_price_bounds(filters)
        )
    )
    candidates: list[UnitCandidate] = []
    for unit_id in sorted(units_by_id):
        if plan_semantics_active and unit_id not in plan_scope_keys:
            continue

        unit, compound = units_by_id[unit_id]
        matching_scope_keys = (
            plan_scope_keys.get(unit_id, set())
            if plan_semantics_active
            else unit_scope_keys[unit_id]
        )
        payment_plans = tuple(
            _to_payment_plan_candidate(payment_plan)
            for payment_plan in plans_by_unit_id.get(unit_id, {}).values()
        )
        candidates.append(
            UnitCandidate(
                unit_id=unit.id,
                external_unit_id=unit.external_unit_id,
                compound_id=compound.id,
                compound_name=compound.name,
                location=compound.location,
                property_type=unit.property_type,
                bedrooms=unit.bedrooms,
                bathrooms=unit.bathrooms,
                area_sqm=unit.area_sqm,
                base_price=unit.base_price,
                base_currency=Currency(unit.base_currency),
                status=UnitStatus(unit.status),
                delivery_date=unit.delivery_date,
                source_last_updated=unit.source_last_updated,
                matched_scope_keys=_ordered_matching_scope_keys(
                    context.resolved_scope_keys,
                    matching_scope_keys,
                ),
                payment_plans=payment_plans,
            )
        )

    return candidates


def _eligible_units_statement(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
    price_basis: PriceBasis,
) -> Select:
    statement = (
        select(Unit, Compound, InventoryScope.scope_key)
        .join(
            Compound,
            and_(
                Compound.company_id == Unit.company_id,
                Compound.id == Unit.compound_id,
            ),
        )
        .join(
            UnitInventoryState,
            and_(
                UnitInventoryState.company_id == Unit.company_id,
                UnitInventoryState.unit_id == Unit.id,
            ),
        )
        .join(
            InventoryScope,
            and_(
                InventoryScope.company_id == UnitInventoryState.company_id,
                InventoryScope.id == UnitInventoryState.inventory_scope_id,
            ),
        )
        .where(
            Unit.company_id == context.company_id,
            Unit.status == UnitStatus.AVAILABLE.value,
            Compound.company_id == context.company_id,
            UnitInventoryState.company_id == context.company_id,
            UnitInventoryState.is_present_in_latest_inventory.is_(True),
            InventoryScope.company_id == context.company_id,
            InventoryScope.scope_key.in_(context.resolved_scope_keys),
        )
        .order_by(Unit.id, InventoryScope.scope_key)
    )

    if filters.location is not None:
        statement = statement.where(Compound.location == filters.location)
    if filters.property_type is not None:
        statement = statement.where(Unit.property_type == filters.property_type)
    if filters.bedrooms is not None:
        statement = statement.where(Unit.bedrooms == filters.bedrooms)
    if filters.bathrooms is not None:
        statement = statement.where(Unit.bathrooms == filters.bathrooms)
    if price_basis is PriceBasis.BASE_PRICE:
        if filters.min_price is not None:
            statement = statement.where(Unit.base_price >= filters.min_price)
        if filters.max_price is not None:
            statement = statement.where(Unit.base_price <= filters.max_price)
    if filters.min_area_sqm is not None:
        statement = statement.where(Unit.area_sqm >= filters.min_area_sqm)
    if filters.max_area_sqm is not None:
        statement = statement.where(Unit.area_sqm <= filters.max_area_sqm)

    return statement


def _eligible_plans_statement(
    *,
    filters: SearchFilters,
    context: SearchExecutionContext,
    unit_ids: tuple[int, ...],
    price_basis: PriceBasis,
) -> Select:
    statement = (
        select(PaymentPlan, InventoryScope.scope_key)
        .join(
            PaymentPlanInventoryState,
            and_(
                PaymentPlanInventoryState.company_id == PaymentPlan.company_id,
                PaymentPlanInventoryState.payment_plan_id == PaymentPlan.id,
            ),
        )
        .join(
            UnitInventoryState,
            and_(
                UnitInventoryState.company_id == PaymentPlan.company_id,
                UnitInventoryState.unit_id == PaymentPlan.unit_id,
                UnitInventoryState.inventory_scope_id
                == PaymentPlanInventoryState.inventory_scope_id,
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
        .where(
            PaymentPlan.company_id == context.company_id,
            PaymentPlan.unit_id.in_(unit_ids),
            PaymentPlanInventoryState.company_id == context.company_id,
            PaymentPlanInventoryState.is_present_in_latest_inventory.is_(True),
            UnitInventoryState.company_id == context.company_id,
            UnitInventoryState.is_present_in_latest_inventory.is_(True),
            InventoryScope.company_id == context.company_id,
            InventoryScope.scope_key.in_(context.resolved_scope_keys),
        )
        .order_by(PaymentPlan.unit_id, PaymentPlan.id, InventoryScope.scope_key)
    )

    if filters.payment_method is not None:
        statement = statement.where(
            PaymentPlan.payment_method == filters.payment_method
        )
    if filters.max_down_payment is not None:
        statement = statement.where(
            PaymentPlan.down_payment <= filters.max_down_payment
        )
    if filters.min_installment_years is not None:
        statement = statement.where(
            PaymentPlan.installment_years >= filters.min_installment_years
        )
    if price_basis is PriceBasis.PLAN_TOTAL_PRICE:
        if filters.min_price is not None:
            statement = statement.where(
                PaymentPlan.total_price >= filters.min_price
            )
        if filters.max_price is not None:
            statement = statement.where(
                PaymentPlan.total_price <= filters.max_price
            )

    return statement


def _has_price_bounds(filters: SearchFilters) -> bool:
    return filters.min_price is not None or filters.max_price is not None


def _ordered_matching_scope_keys(
    resolved_scope_keys: tuple[str, ...],
    matching_scope_keys: set[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered_scope_keys: list[str] = []
    for scope_key in resolved_scope_keys:
        if scope_key in matching_scope_keys and scope_key not in seen:
            seen.add(scope_key)
            ordered_scope_keys.append(scope_key)
    return tuple(ordered_scope_keys)


def _to_payment_plan_candidate(
    payment_plan: PaymentPlan,
) -> PaymentPlanCandidate:
    return PaymentPlanCandidate(
        plan_key=payment_plan.plan_key,
        source_plan_id=payment_plan.source_plan_id,
        plan_name=payment_plan.plan_name,
        payment_method=PaymentMethod(payment_plan.payment_method),
        down_payment=payment_plan.down_payment,
        installment_years=payment_plan.installment_years,
        total_price=payment_plan.total_price,
        currency=Currency(payment_plan.currency),
    )
