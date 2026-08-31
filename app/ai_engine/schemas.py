from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class UnitStatus(str, Enum):
    AVAILABLE = "Available"
    HOLD = "Hold"
    RESERVED = "Reserved"
    SOLD = "Sold"


class UnitSchema(BaseModel):
    """Schema for a real-estate inventory unit."""

    unit_id: str
    compound_name: str
    location: str
    property_type: str
    price: float = Field(gt=0)
    status: UnitStatus
    source: str
    last_updated: date

    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    area_sqm: float | None = Field(default=None, gt=0)
    down_payment: float | None = Field(default=None, ge=0)
    installment_years: float | None = Field(default=None, ge=0)
    delivery_date: date | None = None
