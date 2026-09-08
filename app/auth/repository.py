from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ApiCredential


def find_api_credential_by_key_id(
    session: Session,
    *,
    key_id: str,
) -> ApiCredential | None:
    statement = select(ApiCredential).where(ApiCredential.key_id == key_id)
    return session.execute(statement).scalar_one_or_none()
