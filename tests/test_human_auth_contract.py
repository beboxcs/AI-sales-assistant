import inspect
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone

import pytest

import app.auth.human as human_auth
from app.auth.human import (
    AuthInfrastructureError,
    HumanAuthError,
    InactiveMembershipError,
    InactiveUserError,
    InvalidCredentialsError,
    InvalidSessionError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human.service import (
    authenticate_credentials,
    authenticate_session,
    create_session,
    revoke_session,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)
from app.domain.enums import MembershipRole


PUBLIC_NAMES = [
    "AuthenticatedPrincipal",
    "AuthenticatedUserContext",
    "CreatedSession",
    "HumanAuthError",
    "InvalidCredentialsError",
    "InvalidSessionError",
    "SessionExpiredError",
    "SessionRevokedError",
    "InactiveUserError",
    "InactiveMembershipError",
    "AuthInfrastructureError",
    "authenticate_credentials",
    "create_session",
    "authenticate_session",
    "revoke_session",
]


@pytest.mark.parametrize(
    ("contract_type", "expected_fields"),
    [
        (
            AuthenticatedPrincipal,
            [
                ("user_id", int),
                ("membership_id", int),
                ("company_id", int),
                ("role", MembershipRole),
            ],
        ),
        (
            AuthenticatedUserContext,
            [
                ("user_id", int),
                ("membership_id", int),
                ("company_id", int),
                ("role", MembershipRole),
                ("session_id", int),
            ],
        ),
        (
            CreatedSession,
            [
                ("token", str),
                ("expires_at", datetime),
                ("context", AuthenticatedUserContext),
            ],
        ),
    ],
)
def test_contract_dataclasses_have_exact_fields_and_are_frozen(
    contract_type,
    expected_fields,
):
    assert [(field.name, field.type) for field in fields(contract_type)] == (
        expected_fields
    )

    principal = AuthenticatedPrincipal(1, 2, 3, MembershipRole.SALESPERSON)
    context = AuthenticatedUserContext(
        1,
        2,
        3,
        MembershipRole.SALESPERSON,
        4,
    )
    instances = {
        AuthenticatedPrincipal: principal,
        AuthenticatedUserContext: context,
        CreatedSession: CreatedSession(
            "token",
            datetime(2026, 9, 11, tzinfo=timezone.utc),
            context,
        ),
    }
    with pytest.raises(FrozenInstanceError):
        instances[contract_type].user_id = 99


def test_exception_hierarchy_is_exact():
    assert HumanAuthError.__bases__ == (Exception,)
    assert InvalidCredentialsError.__bases__ == (HumanAuthError,)
    assert InvalidSessionError.__bases__ == (HumanAuthError,)
    assert SessionExpiredError.__bases__ == (InvalidSessionError,)
    assert SessionRevokedError.__bases__ == (InvalidSessionError,)
    assert InactiveUserError.__bases__ == (InvalidSessionError,)
    assert InactiveMembershipError.__bases__ == (InvalidSessionError,)
    assert AuthInfrastructureError.__bases__ == (HumanAuthError,)


@pytest.mark.parametrize(
    ("error_type", "catch_type"),
    [
        (InactiveUserError, InvalidSessionError),
        (InactiveMembershipError, InvalidSessionError),
        (AuthInfrastructureError, HumanAuthError),
    ],
)
def test_new_public_errors_are_catchable_by_their_contract_base(
    error_type,
    catch_type,
):
    with pytest.raises(catch_type):
        raise error_type


def test_service_signatures_are_frozen():
    assert inspect.signature(authenticate_credentials) == inspect.Signature(
        [
            inspect.Parameter("email", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "password",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            ),
        ],
        return_annotation=AuthenticatedPrincipal,
    )
    assert inspect.signature(create_session) == inspect.Signature(
        [
            inspect.Parameter(
                "principal",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=AuthenticatedPrincipal,
            ),
            inspect.Parameter(
                "expires_at",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=datetime,
            ),
        ],
        return_annotation=CreatedSession,
    )
    assert inspect.signature(authenticate_session) == inspect.Signature(
        [
            inspect.Parameter(
                "token",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            ),
        ],
        return_annotation=AuthenticatedUserContext,
    )
    assert inspect.signature(revoke_session) == inspect.Signature(
        [
            inspect.Parameter(
                "session_id",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=int,
            ),
        ],
        return_annotation=None,
    )


@pytest.mark.parametrize(
    ("service_function", "arguments"),
    [
        (authenticate_credentials, ("user@example.com", "secret")),
        (
            create_session,
            (
                AuthenticatedPrincipal(
                    1,
                    2,
                    3,
                    MembershipRole.COMPANY_ADMIN,
                ),
                datetime(2026, 9, 11, tzinfo=timezone.utc),
            ),
        ),
        (authenticate_session, ("token",)),
        (revoke_session, (4,)),
    ],
)
def test_service_placeholders_fail_closed(service_function, arguments):
    with pytest.raises(NotImplementedError):
        service_function(*arguments)


def test_package_reexports_only_the_frozen_public_contract():
    assert human_auth.__all__ == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(human_auth, name) is globals()[name]
