from datetime import datetime
from typing import Protocol

from fastapi import APIRouter, Depends, Header, Request, Security
from pydantic import BaseModel, ConfigDict

from app.api.errors import APIError
from app.api.invitation_delivery import (
    InvitationDeliveryError,
    InvitationDeliveryService,
    get_invitation_delivery_service,
)
from app.api.security import (
    HumanCsrfService,
    HumanSessionResolver,
    get_human_csrf_service,
    get_human_session_resolver,
    require_trusted_origin,
    require_unambiguous_credentials,
    session_cookie,
)
from app.auth import human as human_auth
from app.auth.human.exceptions import (
    AuthInfrastructureError,
    InvitationAuthorizationError,
    InvitationConflictError,
    InvitationInfrastructureError,
    InvalidCsrfTokenError,
    InvalidSessionError,
)
from app.auth.human.types import AuthenticatedUserContext, CreatedInvitation
from app.domain.enums import MembershipRole

router = APIRouter(prefix="/company", tags=["company"])


class CreateInvitationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    role: MembershipRole


class CreateInvitationResponse(BaseModel):
    invitation_id: int
    email: str
    role: MembershipRole
    expires_at: datetime


class InvitationCreationService(Protocol):
    def create_invitation(
        self,
        *,
        company_id: int,
        created_by_user_id: int,
        email: str,
        role: MembershipRole,
    ) -> CreatedInvitation: ...


class PublicInvitationCreationService:
    def create_invitation(
        self,
        *,
        company_id: int,
        created_by_user_id: int,
        email: str,
        role: MembershipRole,
    ) -> CreatedInvitation:
        return human_auth.create_invitation(
            company_id=company_id,
            created_by_user_id=created_by_user_id,
            email=email,
            role=role,
        )


def get_invitation_creation_service() -> InvitationCreationService:
    return PublicInvitationCreationService()


@router.post(
    "/invitations",
    response_model=CreateInvitationResponse,
    status_code=201,
)
def create_company_invitation(
    invitation_request: CreateInvitationRequest,
    request: Request,
    session_token: str | None = Security(session_cookie),
    csrf_token: str | None = Header(default=None, alias="X-CSRF-Token"),
    resolve_human_session: HumanSessionResolver = Depends(
        get_human_session_resolver
    ),
    csrf_service: HumanCsrfService = Depends(get_human_csrf_service),
    invitation_service: InvitationCreationService = Depends(
        get_invitation_creation_service
    ),
    delivery_service: InvitationDeliveryService = Depends(
        get_invitation_delivery_service
    ),
) -> CreateInvitationResponse:
    require_unambiguous_credentials(
        raw_api_key=request.headers.get("x-api-key"),
        session_token=session_token,
    )
    if session_token is None:
        raise _authentication_required_error()

    require_trusted_origin(request)
    if csrf_token is None:
        raise _invalid_csrf_token_error()
    _verify_csrf_token(
        session_token=session_token,
        csrf_token=csrf_token,
        csrf_service=csrf_service,
    )

    current_user = _resolve_current_user(
        session_token=session_token,
        resolve_human_session=resolve_human_session,
    )
    if current_user.role is not MembershipRole.COMPANY_ADMIN:
        raise _forbidden_error()

    try:
        created = invitation_service.create_invitation(
            company_id=current_user.company_id,
            created_by_user_id=current_user.user_id,
            email=invitation_request.email,
            role=invitation_request.role,
        )
    except InvitationAuthorizationError as exc:
        raise _forbidden_error() from exc
    except InvitationConflictError as exc:
        raise _create_invitation_conflict_error() from exc
    except InvitationInfrastructureError as exc:
        raise _authentication_unavailable_error() from exc
    except Exception as exc:
        raise _authentication_unavailable_error() from exc

    try:
        delivery_service.deliver_invitation(
            recipient_email=created.normalized_email,
            token=created.token,
            expires_at=created.expires_at,
            role=created.role,
        )
    except InvitationDeliveryError as exc:
        raise _invitation_delivery_unavailable_error() from exc
    except Exception as exc:
        raise _invitation_delivery_unavailable_error() from exc

    return CreateInvitationResponse(
        invitation_id=created.invitation_id,
        email=created.normalized_email,
        role=created.role,
        expires_at=created.expires_at,
    )


def _resolve_current_user(
    *,
    session_token: str,
    resolve_human_session: HumanSessionResolver,
) -> AuthenticatedUserContext:
    try:
        return resolve_human_session(token=session_token)
    except InvalidSessionError as exc:
        raise _invalid_session_error() from exc
    except AuthInfrastructureError as exc:
        raise _authentication_unavailable_error() from exc
    except Exception as exc:
        raise _authentication_unavailable_error() from exc


def _verify_csrf_token(
    *,
    session_token: str,
    csrf_token: str,
    csrf_service: HumanCsrfService,
) -> None:
    try:
        csrf_service.verify_csrf_token(
            session_token=session_token,
            csrf_token=csrf_token,
        )
    except InvalidCsrfTokenError as exc:
        raise _invalid_csrf_token_error() from exc
    except InvalidSessionError as exc:
        raise _invalid_session_error() from exc
    except AuthInfrastructureError as exc:
        raise _authentication_unavailable_error() from exc
    except Exception as exc:
        raise _authentication_unavailable_error() from exc


def _authentication_required_error() -> APIError:
    return APIError(
        status_code=401,
        code="AUTHENTICATION_REQUIRED",
        message="Authentication is required",
    )


def _invalid_session_error() -> APIError:
    return APIError(
        status_code=401,
        code="INVALID_SESSION",
        message="Invalid or unavailable session",
    )


def _invalid_csrf_token_error() -> APIError:
    return APIError(
        status_code=403,
        code="INVALID_CSRF_TOKEN",
        message="Request verification failed",
    )


def _forbidden_error() -> APIError:
    return APIError(
        status_code=403,
        code="FORBIDDEN",
        message="You do not have permission to perform this action",
    )


def _create_invitation_conflict_error() -> APIError:
    return APIError(
        status_code=409,
        code="INVITATION_CONFLICT",
        message="Invitation could not be created for this account",
    )


def _authentication_unavailable_error() -> APIError:
    return APIError(
        status_code=500,
        code="AUTHENTICATION_UNAVAILABLE",
        message="Authentication could not be completed",
    )


def _invitation_delivery_unavailable_error() -> APIError:
    return APIError(
        status_code=503,
        code="INVITATION_DELIVERY_UNAVAILABLE",
        message="Invitation could not be delivered",
    )
