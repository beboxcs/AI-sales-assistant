"""Controlled values shared across application boundaries."""

from enum import Enum


class UnitStatus(str, Enum):
    AVAILABLE = "Available"
    HOLD = "Hold"
    RESERVED = "Reserved"
    SOLD = "Sold"


class PaymentMethod(str, Enum):
    CASH = "Cash"
    INSTALLMENT = "Installment"


class Currency(str, Enum):
    EGP = "EGP"


class MembershipRole(str, Enum):
    SALESPERSON = "salesperson"
    COMPANY_ADMIN = "company_admin"
