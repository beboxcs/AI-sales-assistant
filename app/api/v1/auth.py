from typing import Protocol

from fastapi import APIRouter, Cookie, Depends, Header, Request, Response, Security
from pydantic import BaseModel, ConfigDict
from starlette.status import HTTP_204_NO_CONTENT

from app.api.errors import APIError
from app.api.security import (
    SESSION_COOKIE_NAME,
    HumanCsrfService,
    clear_session_cookie,
    get_human_csrf_service,
    get_trusted_human_session_token,
    require_trusted_origin,
    require_unambiguous_credentials,
    session_cookie,
    set_session_cookie,
)
from app.auth import human as human_auth
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvitationConflictError,
    InvitationInfrastructureError,
    InvalidCsrfTokenError,
    InvalidCredentialsError,
    InvalidInvitationError,
    InvalidInvitationPasswordError,
    InvalidSessionError,
    LoginRateLimitExceededError,
    LoginRateLimitInfrastructureError,
)
from app.auth.human.service import (
    authenticate_credentials,
    create_session,
)
from app.auth.human.types import (
    AcceptedInvitation,
    AuthenticatedPrincipal,
    CreatedSession,
)
from app.auth.human.types import LoginRateLimitPermit
from app.domain.enums import MembershipRole

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    password: str


class LoginResponse(BaseModel):
    user_id: int
    company_id: int
    role: MembershipRole


class CsrfTokenResponse(BaseModel):
    csrf_token: str


class AcceptInvitationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    password: str


class AcceptInvitationResponse(BaseModel):
    user_id: int
    company_id: int
    membership_id: int
    role: MembershipRole


class HumanAuthService(Protocol):
    def authenticate_credentials(
        self,
        *,
        email: str,
        password: str,
    ) -> AuthenticatedPrincipal: ...

    def create_session(
        self,
        *,
        principal: AuthenticatedPrincipal,
    ) -> CreatedSession: ...


class HumanSessionRevocationService(Protocol):
    def revoke_session(self, *, token: str) -> None: ...


class LoginRateLimitService(Protocol):
    def check_login_rate_limit(
        self,
        *,
        client_ip: str,
        email: str,
    ) -> LoginRateLimitPermit: ...

    def record_login_failure(self, *, permit: LoginRateLimitPermit) -> None: ...

    def record_login_success(self, *, permit: LoginRateLimitPermit) -> None: ...

    def record_login_aborted(self, *, permit: LoginRateLimitPermit) -> None: ...


class InvitationAcceptanceService(Protocol):
    def accept_invitation(
        self,
        *,
        token: str,
        password: str,
    ) -> AcceptedInvitation: ...


class PublicHumanAuthService:
    def authenticate_credentials(
        self,
        *,
        email: str,
        password: str,
    ) -> AuthenticatedPrincipal:
        return authenticate_credentials(email=email, password=password)

    def create_session(
        self,
        *,
        principal: AuthenticatedPrincipal,
    ) -> CreatedSession:
        return create_session(principal=principal)


def get_human_auth_service() -> HumanAuthService:
    return PublicHumanAuthService()


class PublicHumanSessionRevocationService:
    def revoke_session(self, *, token: str) -> None:
        human_auth.revoke_session(token=token)


def get_human_session_revocation_service() -> HumanSessionRevocationService:
    return PublicHumanSessionRevocationService()


class PublicLoginRateLimitService:
    def check_login_rate_limit(
        self,
        *,
        client_ip: str,
        email: str,
    ) -> LoginRateLimitPermit:
        return human_auth.check_login_rate_limit(client_ip=client_ip, email=email)

    def record_login_failure(self, *, permit: LoginRateLimitPermit) -> None:
        human_auth.record_login_failure(permit=permit)

    def record_login_success(self, *, permit: LoginRateLimitPermit) -> None:
        human_auth.record_login_success(permit=permit)

    def record_login_aborted(self, *, permit: LoginRateLimitPermit) -> None:
        human_auth.record_login_aborted(permit=permit)


def get_login_rate_limit_service() -> LoginRateLimitService:
    return PublicLoginRateLimitService()


class PublicInvitationAcceptanceService:
    def accept_invitation(
        self,
        *,
        token: str,
        password: str,
    ) -> AcceptedInvitation:
        return human_auth.accept_invitation(token=token, password=password)


def get_invitation_acceptance_service() -> InvitationAcceptanceService:
    return PublicInvitationAcceptanceService()


@router.post(
    "/login",
    response_model=LoginResponse,
    dependencies=[Depends(require_trusted_origin)],
)
def login(
    request: LoginRequest,
    http_request: Request,
    response: Response,
    human_auth_service: HumanAuthService = Depends(get_human_auth_service),
    rate_limit_service: LoginRateLimitService = Depends(
        get_login_rate_limit_service
    ),
    raw_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> LoginResponse:
    require_unambiguous_credentials(
        raw_api_key=raw_api_key,
        session_token=session_token,
    )

    client_ip = _direct_client_ip(http_request)
    try:
        rate_limit_permit = rate_limit_service.check_login_rate_limit(
            client_ip=client_ip,
            email=request.email,
        )
    except LoginRateLimitExceededError as exc:
        raise _login_rate_limited_error(
            retry_after_seconds=exc.retry_after_seconds
        ) from exc
    except LoginRateLimitInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    try:
        principal = human_auth_service.authenticate_credentials(
            email=request.email,
            password=request.password,
        )
    except InvalidCredentialsError as exc:
        try:
            rate_limit_service.record_login_failure(permit=rate_limit_permit)
        except LoginRateLimitInfrastructureError as record_exc:
            raise _human_auth_unavailable_error() from record_exc
        except Exception as record_exc:
            raise _human_auth_unavailable_error() from record_exc
        raise _invalid_credentials_error() from exc
    except AuthInfrastructureError as exc:
        _record_login_aborted_or_raise(
            rate_limit_service=rate_limit_service,
            permit=rate_limit_permit,
        )
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        _record_login_aborted_or_raise(
            rate_limit_service=rate_limit_service,
            permit=rate_limit_permit,
        )
        raise _human_auth_unavailable_error() from exc

    try:
        rate_limit_service.record_login_success(permit=rate_limit_permit)
    except LoginRateLimitInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    try:
        created_session = human_auth_service.create_session(principal=principal)
    except AuthInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    set_session_cookie(
        response,
        created_session.token,
        expires_at=created_session.expires_at,
    )

    return LoginResponse(
        user_id=created_session.context.user_id,
        company_id=created_session.context.company_id,
        role=created_session.context.role,
    )


@router.post(
    "/csrf",
    response_model=CsrfTokenResponse,
)
def issue_csrf_token(
    session_token: str = Depends(get_trusted_human_session_token),
    csrf_service: HumanCsrfService = Depends(get_human_csrf_service),
) -> CsrfTokenResponse:
    try:
        csrf_token = csrf_service.issue_csrf_token(session_token=session_token)
    except InvalidSessionError as exc:
        raise _invalid_session_error() from exc
    except AuthInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    return CsrfTokenResponse(csrf_token=csrf_token)


@router.post(
    "/invitations/accept",
    response_model=AcceptInvitationResponse,
)
def accept_invitation(
    request: AcceptInvitationRequest,
    http_request: Request,
    invitation_service: InvitationAcceptanceService = Depends(
        get_invitation_acceptance_service
    ),
    raw_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> AcceptInvitationResponse:
    require_unambiguous_credentials(
        raw_api_key=raw_api_key,
        session_token=session_token,
    )
    require_trusted_origin(http_request)

    try:
        accepted = invitation_service.accept_invitation(
            token=request.token,
            password=request.password,
        )
    except InvalidInvitationPasswordError as exc:
        raise _invalid_invitation_password_error() from exc
    except InvalidInvitationError as exc:
        raise _invalid_invitation_error() from exc
    except InvitationConflictError as exc:
        raise _accept_invitation_conflict_error() from exc
    except InvitationInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    return AcceptInvitationResponse(
        user_id=accepted.user_id,
        company_id=accepted.company_id,
        membership_id=accepted.membership_id,
        role=accepted.role,
    )


@router.post(
    "/logout",
    status_code=HTTP_204_NO_CONTENT,
    response_class=Response,
)
def logout(
    request: Request,
    response: Response,
    session_token: str | None = Security(session_cookie),
    csrf_token: str | None = Header(default=None, alias="X-CSRF-Token"),
    csrf_service: HumanCsrfService = Depends(get_human_csrf_service),
    revocation_service: HumanSessionRevocationService = Depends(
        get_human_session_revocation_service
    ),
) -> Response:
    raw_api_key = request.headers.get("x-api-key")
    require_unambiguous_credentials(
        raw_api_key=raw_api_key,
        session_token=session_token,
    )

    if session_token is None:
        if raw_api_key is not None:
            raise _authentication_required_error()
        clear_session_cookie(response)
        response.status_code = HTTP_204_NO_CONTENT
        return response

    require_trusted_origin(request)
    if csrf_token is None:
        raise _invalid_csrf_token_error()

    try:
        csrf_service.verify_csrf_token(
            session_token=session_token,
            csrf_token=csrf_token,
        )
    except InvalidCsrfTokenError as exc:
        raise _invalid_csrf_token_error() from exc
    except InvalidSessionError:
        clear_session_cookie(response)
        response.status_code = HTTP_204_NO_CONTENT
        return response
    except AuthInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    try:
        revocation_service.revoke_session(token=session_token)
    except InvalidSessionError:
        pass
    except AuthInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc

    clear_session_cookie(response)
    response.status_code = HTTP_204_NO_CONTENT
    return response


def _invalid_credentials_error() -> APIError:
    return APIError(
        status_code=401,
        code="INVALID_CREDENTIALS",
        message="Invalid email or password",
    )


def _invalid_session_error() -> APIError:
    return APIError(
        status_code=401,
        code="INVALID_SESSION",
        message="Invalid or unavailable session",
    )


def _authentication_required_error() -> APIError:
    return APIError(
        status_code=401,
        code="AUTHENTICATION_REQUIRED",
        message="Authentication is required",
    )


def _invalid_csrf_token_error() -> APIError:
    return APIError(
        status_code=403,
        code="INVALID_CSRF_TOKEN",
        message="Request verification failed",
    )


def _direct_client_ip(request: Request) -> str:
    client = request.client
    if client is None or not isinstance(client.host, str) or not client.host.strip():
        raise _human_auth_unavailable_error()
    return client.host


def _record_login_aborted_or_raise(
    *,
    rate_limit_service: LoginRateLimitService,
    permit: LoginRateLimitPermit,
) -> None:
    try:
        rate_limit_service.record_login_aborted(permit=permit)
    except LoginRateLimitInfrastructureError as exc:
        raise _human_auth_unavailable_error() from exc
    except Exception as exc:
        raise _human_auth_unavailable_error() from exc


def _login_rate_limited_error(*, retry_after_seconds: int) -> APIError:
    return APIError(
        status_code=429,
        code="LOGIN_RATE_LIMITED",
        message="Too many login attempts. Please try again later.",
        headers={"Retry-After": str(retry_after_seconds)},
    )


def _invalid_invitation_error() -> APIError:
    return APIError(
        status_code=400,
        code="INVALID_INVITATION",
        message="Invitation is invalid or unavailable",
    )


def _invalid_invitation_password_error() -> APIError:
    return APIError(
        status_code=422,
        code="INVALID_INVITATION_PASSWORD",
        message="Invitation password is invalid",
    )


def _accept_invitation_conflict_error() -> APIError:
    return APIError(
        status_code=409,
        code="INVITATION_CONFLICT",
        message="Invitation could not be accepted",
    )


def _human_auth_unavailable_error() -> APIError:
    return APIError(
        status_code=500,
        code="AUTHENTICATION_UNAVAILABLE",
        message="Authentication could not be completed",
    )
