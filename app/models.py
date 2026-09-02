from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    and_,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from app.database import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    compounds: Mapped[list[Compound]] = relationship(back_populates="company")
    inventory_scopes: Mapped[list[InventoryScope]] = relationship(
        back_populates="company", foreign_keys="InventoryScope.company_id"
    )


class Compound(Base):
    __tablename__ = "compounds"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "id", name="uq_compounds_company_id_id"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    company: Mapped[Company] = relationship(back_populates="compounds")
    inventory_scopes: Mapped[list[InventoryScope]] = relationship(
        back_populates="compound",
        primaryjoin=lambda: and_(
            Compound.company_id == InventoryScope.company_id,
            Compound.id == foreign(InventoryScope.compound_id),
        ),
    )
    units: Mapped[list[Unit]] = relationship(back_populates="compound")


class InventoryScope(Base):
    __tablename__ = "inventory_scopes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "compound_id"],
            ["compounds.company_id", "compounds.id"],
            name="fk_inventory_scopes_company_compound",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "company_id", "id", name="uq_inventory_scopes_company_id_id"
        ),
        UniqueConstraint(
            "company_id", "scope_key", name="uq_inventory_scopes_company_scope_key"
        ),
        Index("ix_inventory_scopes_company_id", "company_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="RESTRICT"), nullable=False
    )
    compound_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scope_key: Mapped[str] = mapped_column(String, nullable=False)
    source_identifier: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    company: Mapped[Company] = relationship(
        back_populates="inventory_scopes", foreign_keys=[company_id]
    )
    compound: Mapped[Compound | None] = relationship(
        back_populates="inventory_scopes",
        primaryjoin=lambda: and_(
            Compound.company_id == InventoryScope.company_id,
            Compound.id == foreign(InventoryScope.compound_id),
        ),
    )
    unit_inventory_states: Mapped[list[UnitInventoryState]] = relationship(
        back_populates="inventory_scope",
        primaryjoin=lambda: and_(
            InventoryScope.company_id == UnitInventoryState.company_id,
            InventoryScope.id == foreign(UnitInventoryState.inventory_scope_id),
        ),
    )
    payment_plan_inventory_states: Mapped[list[PaymentPlanInventoryState]] = (
        relationship(
            back_populates="inventory_scope",
            primaryjoin=lambda: and_(
                InventoryScope.company_id == PaymentPlanInventoryState.company_id,
                InventoryScope.id
                == foreign(PaymentPlanInventoryState.inventory_scope_id),
            ),
        )
    )
    unit_history_entries: Mapped[list[UnitHistory]] = relationship(
        back_populates="inventory_scope",
        primaryjoin=lambda: and_(
            InventoryScope.company_id == UnitHistory.company_id,
            InventoryScope.id == foreign(UnitHistory.inventory_scope_id),
        ),
    )


class Unit(Base):
    __tablename__ = "units"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "compound_id"],
            ["compounds.company_id", "compounds.id"],
            name="fk_units_company_compound",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("company_id", "id", name="uq_units_company_id_id"),
        UniqueConstraint(
            "company_id",
            "compound_id",
            "external_unit_id",
            name="uq_units_company_compound_external_unit",
        ),
        CheckConstraint(
            "base_price IS NULL OR base_price > 0",
            name="ck_units_base_price_positive",
        ),
        CheckConstraint(
            "base_currency = 'EGP'", name="ck_units_base_currency_egp"
        ),
        CheckConstraint(
            "status IN ('Available', 'Hold', 'Reserved', 'Sold')",
            name="ck_units_status_allowed",
        ),
        CheckConstraint(
            "bedrooms IS NULL OR bedrooms >= 0",
            name="ck_units_bedrooms_nonnegative",
        ),
        CheckConstraint(
            "bathrooms IS NULL OR bathrooms >= 0",
            name="ck_units_bathrooms_nonnegative",
        ),
        CheckConstraint(
            "area_sqm IS NULL OR area_sqm > 0",
            name="ck_units_area_sqm_positive",
        ),
        Index("ix_units_company_id", "company_id"),
        Index("ix_units_company_id_status", "company_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False)
    compound_id: Mapped[int] = mapped_column(Integer, nullable=False)
    external_unit_id: Mapped[str] = mapped_column(String, nullable=False)
    property_type: Mapped[str] = mapped_column(String, nullable=False)
    base_price: Mapped[Decimal | None] = mapped_column(
        Numeric(15, 2), nullable=True
    )
    base_currency: Mapped[str] = mapped_column(
        String, nullable=False, default="EGP", server_default=text("'EGP'")
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    bedrooms: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    bathrooms: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    area_sqm: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    delivery_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_last_updated: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    compound: Mapped[Compound] = relationship(back_populates="units")
    payment_plans: Mapped[list[PaymentPlan]] = relationship(back_populates="unit")
    inventory_states: Mapped[list[UnitInventoryState]] = relationship(
        back_populates="unit"
    )
    history_entries: Mapped[list[UnitHistory]] = relationship(back_populates="unit")


class PaymentPlan(Base):
    __tablename__ = "payment_plans"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "unit_id"],
            ["units.company_id", "units.id"],
            name="fk_payment_plans_company_unit",
        ),
        UniqueConstraint(
            "company_id", "id", name="uq_payment_plans_company_id_id"
        ),
        UniqueConstraint("unit_id", "plan_key", name="uq_payment_plans_unit_plan_key"),
        CheckConstraint(
            "payment_method IN ('Cash', 'Installment')",
            name="ck_payment_plans_payment_method_allowed",
        ),
        CheckConstraint(
            "payment_method != 'Cash' OR installment_years IS NULL",
            name="ck_payment_plans_cash_installment_years_null",
        ),
        CheckConstraint(
            "down_payment IS NULL OR down_payment >= 0",
            name="ck_payment_plans_down_payment_nonnegative",
        ),
        CheckConstraint(
            "installment_years IS NULL OR installment_years > 0",
            name="ck_payment_plans_installment_years_positive",
        ),
        CheckConstraint(
            "total_price IS NULL OR total_price > 0",
            name="ck_payment_plans_total_price_positive",
        ),
        CheckConstraint("currency = 'EGP'", name="ck_payment_plans_currency_egp"),
        Index("ix_payment_plans_unit_id", "unit_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_id: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_key: Mapped[str] = mapped_column(String, nullable=False)
    source_plan_id: Mapped[str | None] = mapped_column(String, nullable=True)
    plan_name: Mapped[str | None] = mapped_column(String, nullable=True)
    payment_method: Mapped[str] = mapped_column(String, nullable=False)
    down_payment: Mapped[Decimal | None] = mapped_column(
        Numeric(15, 2), nullable=True
    )
    installment_years: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2), nullable=True
    )
    total_price: Mapped[Decimal | None] = mapped_column(
        Numeric(15, 2), nullable=True
    )
    currency: Mapped[str] = mapped_column(
        String, nullable=False, default="EGP", server_default=text("'EGP'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    unit: Mapped[Unit] = relationship(back_populates="payment_plans")
    payment_inventory_states: Mapped[list[PaymentPlanInventoryState]] = relationship(
        back_populates="payment_plan"
    )


class UnitInventoryState(Base):
    __tablename__ = "unit_inventory_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "unit_id"],
            ["units.company_id", "units.id"],
            name="fk_unit_inventory_states_company_unit",
        ),
        ForeignKeyConstraint(
            ["company_id", "inventory_scope_id"],
            ["inventory_scopes.company_id", "inventory_scopes.id"],
            name="fk_unit_inventory_states_company_scope",
        ),
        UniqueConstraint(
            "unit_id",
            "inventory_scope_id",
            name="uq_unit_inventory_states_unit_scope",
        ),
        Index(
            "ix_unit_inventory_states_inventory_scope_id", "inventory_scope_id"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_id: Mapped[int] = mapped_column(Integer, nullable=False)
    inventory_scope_id: Mapped[int] = mapped_column(Integer, nullable=False)
    is_present_in_latest_inventory: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    unit: Mapped[Unit] = relationship(back_populates="inventory_states")
    inventory_scope: Mapped[InventoryScope] = relationship(
        back_populates="unit_inventory_states",
        primaryjoin=lambda: and_(
            InventoryScope.company_id == UnitInventoryState.company_id,
            InventoryScope.id == foreign(UnitInventoryState.inventory_scope_id),
        ),
    )


class PaymentPlanInventoryState(Base):
    """Tracks payment-plan presence independently within each inventory scope.

    A missing plan is not cancelled. Persistence services will mark a missing plan
    absent in its scope; when its unit is missing, they will mark both the unit and
    all of that unit's known plans absent for the same scope in one transaction.
    Reappearance restores presence and updates ``last_seen_at``. Searches must
    eventually require both unit and payment-plan presence.
    """

    __tablename__ = "payment_plan_inventory_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "payment_plan_id"],
            ["payment_plans.company_id", "payment_plans.id"],
            name="fk_payment_plan_inventory_states_company_plan",
        ),
        ForeignKeyConstraint(
            ["company_id", "inventory_scope_id"],
            ["inventory_scopes.company_id", "inventory_scopes.id"],
            name="fk_payment_plan_inventory_states_company_scope",
        ),
        UniqueConstraint(
            "payment_plan_id",
            "inventory_scope_id",
            name="uq_payment_plan_inventory_states_plan_scope",
        ),
        Index(
            "ix_payment_plan_inventory_states_inventory_scope_id",
            "inventory_scope_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False)
    payment_plan_id: Mapped[int] = mapped_column(Integer, nullable=False)
    inventory_scope_id: Mapped[int] = mapped_column(Integer, nullable=False)
    is_present_in_latest_inventory: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    payment_plan: Mapped[PaymentPlan] = relationship(
        back_populates="payment_inventory_states"
    )
    inventory_scope: Mapped[InventoryScope] = relationship(
        back_populates="payment_plan_inventory_states",
        primaryjoin=lambda: and_(
            InventoryScope.company_id == PaymentPlanInventoryState.company_id,
            InventoryScope.id
            == foreign(PaymentPlanInventoryState.inventory_scope_id),
        ),
    )


class UnitHistory(Base):
    __tablename__ = "unit_history"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "unit_id"],
            ["units.company_id", "units.id"],
            name="fk_unit_history_company_unit",
        ),
        ForeignKeyConstraint(
            ["company_id", "inventory_scope_id"],
            ["inventory_scopes.company_id", "inventory_scopes.id"],
            name="fk_unit_history_company_scope",
        ),
        CheckConstraint(
            "change_type IN ('added', 'updated', 'missing')",
            name="ck_unit_history_change_type_allowed",
        ),
        Index("ix_unit_history_unit_id_recorded_at", "unit_id", "recorded_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_id: Mapped[int] = mapped_column(Integer, nullable=False)
    inventory_scope_id: Mapped[int] = mapped_column(Integer, nullable=False)
    change_type: Mapped[str] = mapped_column(String, nullable=False)
    changes: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    payment_plan_changes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    unit: Mapped[Unit] = relationship(back_populates="history_entries")
    inventory_scope: Mapped[InventoryScope] = relationship(
        back_populates="unit_history_entries",
        primaryjoin=lambda: and_(
            InventoryScope.company_id == UnitHistory.company_id,
            InventoryScope.id == foreign(UnitHistory.inventory_scope_id),
        ),
    )
