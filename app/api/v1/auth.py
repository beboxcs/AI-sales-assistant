from typing import Protocol

from fastapi import APIRouter, Cookie, Depends, Header, Response
from pydantic import BaseModel, ConfigDict

from app.api.errors import APIError
from app.api.security import (
    SESSION_COOKIE_NAME,
    HumanCsrfService,
    get_human_csrf_service,
    get_trusted_human_session_token,
    require_unambiguous_credentials,
    require_trusted_origin,
    set_session_cookie,
)
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvalidCredentialsError,
    InvalidSessionError,
)
from app.auth.human.service import (
    authenticate_credentials,
    create_session,
)
from app.auth.human.types import AuthenticatedPrincipal, CreatedSession
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


@router.post(
    "/login",
    response_model=LoginResponse,
    dependencies=[Depends(require_trusted_origin)],
)
def login(
    request: LoginRequest,
    response: Response,
    human_auth_service: HumanAuthService = Depends(get_human_auth_service),
    raw_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> LoginResponse:
    require_unambiguous_credentials(
        raw_api_key=raw_api_key,
        session_token=session_token,
    )

    try:
        principal = human_auth_service.authenticate_credentials(
            email=request.email,
            password=request.password,
        )
        created_session = human_auth_service.create_session(principal=principal)
    except InvalidCredentialsError as exc:
        raise _invalid_credentials_error() from exc
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


def _human_auth_unavailable_error() -> APIError:
    return APIError(
        status_code=500,
        code="AUTHENTICATION_UNAVAILABLE",
        message="Authentication could not be completed",
    )
