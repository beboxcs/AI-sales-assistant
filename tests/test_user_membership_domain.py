import pytest

from app.domain.enums import MembershipRole
from app.domain.users import normalize_email
from app.models import User


def test_membership_role_contract_contains_only_frozen_values():
    assert {role.value for role in MembershipRole} == {
        "salesperson",
        "company_admin",
    }
    with pytest.raises(ValueError):
        MembershipRole("owner")


def test_normalize_email_trims_and_uses_unicode_casefold():
    assert normalize_email(" \tStraße@EXAMPLE.COM\n") == "strasse@example.com"


def test_normalize_email_does_not_apply_provider_specific_rewriting():
    assert (
        normalize_email(" Hazem.Example+Sales@GMAIL.COM ")
        == "hazem.example+sales@gmail.com"
    )


def test_user_normalizes_email_at_the_orm_boundary():
    user = User(
        normalized_email="  User@Example.COM ",
        password_hash="$argon2id$opaque",
    )

    assert user.normalized_email == "user@example.com"

    user.normalized_email = "  NewUser@Example.COM "

    assert user.normalized_email == "newuser@example.com"


def test_user_orm_boundary_rejects_blank_email():
    with pytest.raises(ValueError, match="Email must not be blank"):
        User(normalized_email=" \t\n ", password_hash="$argon2id$opaque")


@pytest.mark.parametrize("email", ["", "   ", "\t", "\n", " \t\n "])
def test_normalize_email_rejects_blank_or_whitespace_only_values(email):
    with pytest.raises(ValueError, match="Email must not be blank"):
        normalize_email(email)
