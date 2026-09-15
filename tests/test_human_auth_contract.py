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
    InvalidCsrfTokenError,
    InvalidSessionError,
    LoginRateLimitExceededError,
    LoginRateLimitInfrastructureError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human.login_rate_limit import (
    check_login_rate_limit,
    record_login_aborted,
    record_login_failure,
    record_login_success,
)
from app.auth.human.service import (
    authenticate_credentials,
    create_session,
    issue_csrf_token,
    resolve_session,
    revoke_session,
    verify_csrf_token,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
    LoginRateLimitPermit,
)
from app.domain.enums import MembershipRole


PUBLIC_NAMES = [
    "AuthenticatedPrincipal",
    "AuthenticatedUserContext",
    "CreatedSession",
    "LoginRateLimitPermit",
    "HumanAuthError",
    "InvalidCredentialsError",
    "InvalidCsrfTokenError",
    "InvalidSessionError",
    "LoginRateLimitExceededError",
    "LoginRateLimitInfrastructureError",
    "SessionExpiredError",
    "SessionRevokedError",
    "InactiveUserError",
    "InactiveMembershipError",
    "AuthInfrastructureError",
    "authenticate_credentials",
    "create_session",
    "issue_csrf_token",
    "resolve_session",
    "revoke_session",
    "verify_csrf_token",
    "check_login_rate_limit",
    "record_login_aborted",
    "record_login_failure",
    "record_login_success",
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
        (
            LoginRateLimitPermit,
            [("attempt_id", str)],
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
        LoginRateLimitPermit: LoginRateLimitPermit("attempt-id"),
    }
    with pytest.raises(FrozenInstanceError):
        instances[contract_type].user_id = 99


def test_exception_hierarchy_is_exact():
    assert HumanAuthError.__bases__ == (Exception,)
    assert InvalidCredentialsError.__bases__ == (HumanAuthError,)
    assert InvalidCsrfTokenError.__bases__ == (HumanAuthError,)
    assert InvalidSessionError.__bases__ == (HumanAuthError,)
    assert LoginRateLimitExceededError.__bases__ == (HumanAuthError,)
    assert LoginRateLimitInfrastructureError.__bases__ == (HumanAuthError,)
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
        (InvalidCsrfTokenError, HumanAuthError),
        (LoginRateLimitInfrastructureError, HumanAuthError),
        (AuthInfrastructureError, HumanAuthError),
    ],
)
def test_new_public_errors_are_catchable_by_their_contract_base(
    error_type,
    catch_type,
):
    with pytest.raises(catch_type):
        raise error_type


def test_login_rate_limit_exceeded_error_is_catchable_by_human_auth_error():
    with pytest.raises(HumanAuthError):
        raise LoginRateLimitExceededError(retry_after_seconds=1)


def test_service_signatures_are_frozen():
    assert inspect.signature(authenticate_credentials) == inspect.Signature(
        [
            inspect.Parameter(
                "email",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
            inspect.Parameter(
                "password",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
        ],
        return_annotation=AuthenticatedPrincipal,
    )
    assert inspect.signature(create_session) == inspect.Signature(
        [
            inspect.Parameter(
                "principal",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=AuthenticatedPrincipal,
            ),
        ],
        return_annotation=CreatedSession,
    )
    assert inspect.signature(resolve_session) == inspect.Signature(
        [
            inspect.Parameter(
                "token",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
        ],
        return_annotation=AuthenticatedUserContext,
    )
    assert inspect.signature(revoke_session) == inspect.Signature(
        [
            inspect.Parameter(
                "token",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
        ],
        return_annotation=None,
    )
    assert inspect.signature(issue_csrf_token) == inspect.Signature(
        [
            inspect.Parameter(
                "session_token",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
        ],
        return_annotation=str,
    )
    assert inspect.signature(verify_csrf_token) == inspect.Signature(
        [
            inspect.Parameter(
                "session_token",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
            inspect.Parameter(
                "csrf_token",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
        ],
        return_annotation=None,
    )
    assert inspect.signature(check_login_rate_limit) == inspect.Signature(
        [
            inspect.Parameter(
                "client_ip",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
            inspect.Parameter(
                "email",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            ),
        ],
        return_annotation=LoginRateLimitPermit,
    )
    assert inspect.signature(record_login_failure) == inspect.Signature(
        [
            inspect.Parameter(
                "permit",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=LoginRateLimitPermit,
            ),
        ],
        return_annotation=None,
    )
    assert inspect.signature(record_login_aborted) == inspect.Signature(
        [
            inspect.Parameter(
                "permit",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=LoginRateLimitPermit,
            ),
        ],
        return_annotation=None,
    )
    assert inspect.signature(record_login_success) == inspect.Signature(
        [
            inspect.Parameter(
                "permit",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=LoginRateLimitPermit,
            ),
        ],
        return_annotation=None,
    )


def test_create_session_rejects_caller_supplied_expiration():
    principal = AuthenticatedPrincipal(
        1,
        2,
        3,
        MembershipRole.COMPANY_ADMIN,
    )

    with pytest.raises(TypeError):
        create_session(
            principal=principal,
            expires_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        )


def test_package_reexports_only_the_frozen_public_contract():
    assert human_auth.__all__ == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(human_auth, name) is globals()[name]
