from __future__ import annotations

import hashlib
import inspect
import threading
import time
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.human import (
    AuthInfrastructureError, CreatedPasswordReset, InvalidCredentialsError,
    InvalidPasswordResetPasswordError, InvalidPasswordResetTokenError,
    PasswordResetInfrastructureError, PasswordResetResult, SessionRevokedError,
    StaleCredentialsError, request_password_reset, reset_password,
)
from app.auth.human import password_reset_service as resets
from app.auth.human import service as humans
from app.auth.human.passwords import hash_password, verify_password
from app.auth.human.sessions import digest_session_token
from app.database import Base, engine
from app.domain.enums import MembershipRole
from app.models import (
    ApiCredential, Company, CompanyMembership, PasswordResetToken, User, UserSession,
)

OLD_PASSWORD = "old password"
NEW_PASSWORD = "new password"
OLD_HASH = hash_password(OLD_PASSWORD)
EMAIL = "reset@example.com"


@pytest.fixture
def reset_db(monkeypatch):
    assert engine.dialect.name == "postgresql"
    schema = f"password_reset_{uuid4().hex}"
    quoted = engine.dialect.identifier_preparer.quote(schema)
    with engine.begin() as connection:
        connection.execute(sa.text(f"CREATE SCHEMA {quoted}"))
    isolated = sa.create_engine(
        engine.url,
        connect_args={"options": f"-csearch_path={schema} -clock_timeout=10000 -cstatement_timeout=15000"},
        poolclass=NullPool,
    )
    Base.metadata.create_all(isolated, tables=[
        Company.__table__, User.__table__, CompanyMembership.__table__,
        UserSession.__table__, ApiCredential.__table__, PasswordResetToken.__table__,
    ])
    factory = sessionmaker(bind=isolated)
    monkeypatch.setattr(resets, "SessionLocal", factory)
    monkeypatch.setattr(humans, "SessionLocal", factory)
    try:
        yield isolated
    finally:
        isolated.dispose()
        with engine.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA {quoted} CASCADE"))


@pytest.fixture
def identity(reset_db):
    with Session(reset_db) as session:
        company = Company(name="Reset company")
        user = User(normalized_email=EMAIL, password_hash=OLD_HASH)
        session.add_all([company, user])
        session.flush()
        membership = CompanyMembership(
            user_id=user.id, company_id=company.id,
            role=MembershipRole.SALESPERSON.value,
        )
        session.add(membership)
        session.flush()
        ids = user.id, membership.id, company.id
        session.commit()
    return ids


def authenticate(*, company_id: int):
    return humans.authenticate_credentials(
        email=EMAIL,
        password=OLD_PASSWORD,
        company_id=company_id,
    )


def snapshot(database):
    with Session(database) as session:
        users = [tuple(row) for row in session.execute(sa.select(
            User.id, User.password_hash, User.password_version, User.password_changed_at,
        ).order_by(User.id))]
        tokens = [tuple(row) for row in session.execute(sa.select(
            PasswordResetToken.id, PasswordResetToken.token_digest,
            PasswordResetToken.consumed_at, PasswordResetToken.revoked_at,
        ).order_by(PasswordResetToken.id))]
        sessions = [tuple(row) for row in session.execute(sa.select(
            UserSession.id, UserSession.revoked_at, UserSession.last_activity_at,
            UserSession.expires_at,
        ).order_by(UserSession.id))]
    return users, tokens, sessions


def test_public_reset_contract():
    for function, names, return_type in (
        (request_password_reset, ["email"], CreatedPasswordReset | None),
        (reset_password, ["token", "new_password"], PasswordResetResult),
    ):
        signature = inspect.signature(function, eval_str=True)
        assert list(signature.parameters) == names
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.annotation is str
                   and p.default is inspect.Parameter.empty for p in signature.parameters.values())
        assert signature.return_annotation == return_type
    assert [(f.name, f.type) for f in fields(CreatedPasswordReset)] == [
        ("token", str), ("normalized_email", str), ("expires_at", datetime),
    ]
    assert [(f.name, f.type) for f in fields(PasswordResetResult)] == [("user_id", int)]
    for instance in (CreatedPasswordReset("token", EMAIL, datetime.now(timezone.utc)), PasswordResetResult(1)):
        with pytest.raises(FrozenInstanceError):
            instance.user_id = 99


@pytest.mark.parametrize("email", ["unknown@example.com", "", " \t", None])
def test_unknown_or_ineligible_identity_returns_none(reset_db, identity, email):
    before = snapshot(reset_db)
    assert request_password_reset(email=email) is None
    assert snapshot(reset_db) == before


def test_inactive_user_request_and_confirmation_are_ineligible(reset_db, identity):
    created = request_password_reset(email=EMAIL)
    with Session(reset_db) as session:
        session.get(User, identity[0]).is_active = False
        session.commit()
    before = snapshot(reset_db)
    assert request_password_reset(email=EMAIL) is None
    with pytest.raises(InvalidPasswordResetTokenError, match="^Invalid password reset token$"):
        reset_password(token=created.token, new_password=NEW_PASSWORD)
    assert snapshot(reset_db) == before


def test_normalization_entropy_digest_and_exact_expiration(monkeypatch, reset_db, identity):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(resets, "_utc_now", lambda: now)
    created = request_password_reset(email=" \tRESET@Example.COM ")
    assert created.normalized_email == EMAIL
    assert len(created.token) == 43
    assert resets.PASSWORD_RESET_TOKEN_ENTROPY_BYTES == 32
    assert created.expires_at == now + timedelta(minutes=30)
    with Session(reset_db) as session:
        row = session.scalar(sa.select(PasswordResetToken))
        assert row.user_id == identity[0]
        assert row.created_at == now
        assert row.expires_at == created.expires_at
        assert row.token_digest == hashlib.sha256(created.token.encode("ascii")).digest()
        assert len(row.token_digest) == 32
        assert created.token.encode("ascii") != row.token_digest
        assert set(row.__table__.columns.keys()) == {
            "id", "user_id", "token_digest", "created_at", "expires_at", "consumed_at", "revoked_at",
        }
        assert row.consumed_at is row.revoked_at is None


def test_rotation_revokes_even_expired_open_rows(monkeypatch, reset_db, identity):
    old = request_password_reset(email=EMAIL)
    monkeypatch.setattr(resets, "_utc_now", lambda: old.expires_at)
    new = request_password_reset(email=EMAIL)
    assert new.token != old.token
    with pytest.raises(InvalidPasswordResetTokenError, match="^Invalid password reset token$"):
        reset_password(token=old.token, new_password=NEW_PASSWORD)
    with Session(reset_db) as session:
        rows = list(session.scalars(sa.select(PasswordResetToken).order_by(PasswordResetToken.id)))
        assert len(rows) == 2
        assert rows[0].revoked_at == old.expires_at
        assert rows[1].revoked_at is rows[1].consumed_at is None
    assert reset_password(token=new.token, new_password=NEW_PASSWORD).user_id == identity[0]


@pytest.mark.parametrize("token", [None, "", " ", "short", "!" * 43, "ü" * 43, "Z" * 43])
def test_malformed_and_unknown_tokens_are_generic_and_do_not_write(reset_db, identity, token):
    before = snapshot(reset_db)
    with pytest.raises(InvalidPasswordResetTokenError, match="^Invalid password reset token$"):
        reset_password(token=token, new_password=NEW_PASSWORD)
    assert snapshot(reset_db) == before


@pytest.mark.parametrize("state", ["expired", "boundary", "consumed", "revoked"])
def test_all_invalid_token_states_are_generic(monkeypatch, reset_db, identity, state):
    created = request_password_reset(email=EMAIL)
    with Session(reset_db) as session:
        row = session.scalar(sa.select(PasswordResetToken))
        if state == "consumed":
            row.consumed_at = row.created_at
        elif state == "revoked":
            row.revoked_at = row.created_at
        session.commit()
    if state in {"expired", "boundary"}:
        now = created.expires_at + (timedelta(seconds=1) if state == "expired" else timedelta())
        monkeypatch.setattr(resets, "_utc_now", lambda: now)
    before = snapshot(reset_db)
    with pytest.raises(InvalidPasswordResetTokenError, match="^Invalid password reset token$"):
        reset_password(token=created.token, new_password=NEW_PASSWORD)
    assert snapshot(reset_db) == before


@pytest.mark.parametrize("password", [None, "", " \t\n"])
def test_existing_password_policy_failure_rolls_back(reset_db, identity, password):
    humans.create_session(principal=authenticate(company_id=identity[2]))
    created = request_password_reset(email=EMAIL)
    before = snapshot(reset_db)
    with pytest.raises(InvalidPasswordResetPasswordError, match="^Invalid password reset password$"):
        reset_password(token=created.token, new_password=password)
    assert snapshot(reset_db) == before


def test_reset_freshness_replay_and_global_revocation(reset_db, identity):
    old_principal = authenticate(company_id=identity[2])
    assert old_principal.password_version == 1
    existing = humans.create_session(principal=old_principal)
    other_token = "other-user-session"
    already_revoked_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    with Session(reset_db) as session:
        second_company = Company(name="Second company")
        unrelated_user = User(normalized_email="unrelated@example.com", password_hash=OLD_HASH)
        session.add_all([second_company, unrelated_user])
        session.flush()
        memberships = [
            CompanyMembership(user_id=identity[0], company_id=second_company.id, role="company_admin"),
            CompanyMembership(user_id=unrelated_user.id, company_id=second_company.id, role="salesperson"),
        ]
        session.add_all(memberships)
        session.flush()
        now = datetime.now(timezone.utc)
        for member, token, revoked in (
            (memberships[0], "second-company-session", None),
            (memberships[0], "already-revoked", already_revoked_at),
            (memberships[1], other_token, None),
        ):
            session.add(UserSession(
                user_id=member.user_id, membership_id=member.id, company_id=member.company_id,
                token_digest=digest_session_token(token=token), created_at=now - timedelta(hours=1),
                expires_at=now + timedelta(hours=7), last_activity_at=now, revoked_at=revoked,
            ))
        credential = ApiCredential(
            key_id="untouched-b1", company_id=identity[2], secret_hash=b"s" * 32,
            name="Unchanged API credential",
        )
        session.add(credential)
        second_membership_id = memberships[0].id
        session.commit()
    with Session(reset_db) as session:
        b1_before = tuple(session.execute(sa.select(ApiCredential.__table__)).one())
        count_before = session.scalar(sa.select(sa.func.count(UserSession.id)))
    created = request_password_reset(email=EMAIL)
    result = reset_password(token=created.token, new_password=NEW_PASSWORD)
    assert result == PasswordResetResult(identity[0])
    with Session(reset_db) as session:
        user = session.get(User, identity[0])
        assert user.password_version == 2
        assert user.password_hash.startswith("$argon2id$")
        assert user.password_hash != OLD_HASH
        assert verify_password(NEW_PASSWORD, user.password_hash)
        row = session.scalar(sa.select(PasswordResetToken))
        assert row.consumed_at == user.password_changed_at
        assert row.revoked_at is None
        for item in session.scalars(sa.select(UserSession)):
            if item.user_id != user.id:
                assert item.revoked_at is None
            elif item.token_digest == digest_session_token(token="already-revoked"):
                assert item.revoked_at == already_revoked_at
            else:
                assert item.revoked_at == row.consumed_at
        assert session.scalar(sa.select(sa.func.count(UserSession.id))) == count_before
        assert tuple(session.execute(sa.select(ApiCredential.__table__)).one()) == b1_before
        # Restore the original one-active-membership login situation only in this test.
        session.get(CompanyMembership, second_membership_id).is_active = False
        session.commit()
    with pytest.raises(SessionRevokedError):
        humans.resolve_session(token=existing.token)
    before = snapshot(reset_db)
    with pytest.raises(StaleCredentialsError, match="^Human authentication unavailable$"):
        humans.create_session(principal=old_principal)
    with pytest.raises(InvalidPasswordResetTokenError):
        reset_password(token=created.token, new_password="another password")
    assert snapshot(reset_db) == before
    with pytest.raises(InvalidCredentialsError):
        authenticate(company_id=identity[2])
    new_principal = humans.authenticate_credentials(
        email=EMAIL,
        password=NEW_PASSWORD,
        company_id=identity[2],
    )
    assert new_principal.password_version == 2
    fresh_session = humans.create_session(principal=new_principal)
    assert humans.resolve_session(token=fresh_session.token).user_id == identity[0]


@pytest.mark.parametrize("fault", ["hash", "hash_value_error", "flush", "revoke", "commit"])
def test_partial_reset_failures_roll_back_everything(monkeypatch, reset_db, identity, fault):
    humans.create_session(principal=authenticate(company_id=identity[2]))
    created = request_password_reset(email=EMAIL)
    before = snapshot(reset_db)

    def fail(*args, **kwargs):
        raise RuntimeError(f"private postgresql://secret {created.token} {NEW_PASSWORD}")

    if fault == "hash_value_error":
        def fail_value_error(*args, **kwargs):
            raise ValueError("private hash backend failure")

        monkeypatch.setattr(resets, "hash_password", fail_value_error)
    elif fault == "hash":
        monkeypatch.setattr(resets, "hash_password", fail)
    elif fault == "revoke":
        real_revoke = resets.revoke_all_user_sessions

        def fail_after_revoke(*args, **kwargs):
            real_revoke(*args, **kwargs)
            fail()

        monkeypatch.setattr(resets, "revoke_all_user_sessions", fail_after_revoke)
    else:
        class FailingSession(Session):
            def flush(self, objects=None):
                result = super().flush(objects)
                if fault == "flush" and self.info.get("reset_write"):
                    fail()
                if any(isinstance(row, PasswordResetToken) and row.consumed_at is not None
                       for row in self.identity_map.values()):
                    self.info["reset_write"] = True
                    if fault == "flush":
                        fail()
                return result

        if fault == "commit":
            sa.event.listen(FailingSession, "before_commit", fail)
        monkeypatch.setattr(resets, "SessionLocal", sessionmaker(bind=reset_db, class_=FailingSession))
    with pytest.raises(PasswordResetInfrastructureError) as error:
        reset_password(token=created.token, new_password=NEW_PASSWORD)
    assert str(error.value) == "Password reset unavailable"
    assert error.value.__suppress_context__
    assert snapshot(reset_db) == before


def test_request_rotation_failure_restores_previous_token(monkeypatch, reset_db, identity):
    old = request_password_reset(email=EMAIL)
    before = snapshot(reset_db)

    def fail(*args, **kwargs):
        raise RuntimeError("private database details")

    monkeypatch.setattr(resets, "persist_reset_token", fail)
    with pytest.raises(PasswordResetInfrastructureError, match="^Password reset unavailable$"):
        request_password_reset(email=EMAIL)
    assert snapshot(reset_db) == before
    assert reset_password(token=old.token, new_password=NEW_PASSWORD).user_id == identity[0]


@pytest.mark.parametrize("mutation", ["inactive_user", "inactive_membership", "company", "role", "user", "invalid_role", "invalid_version", "persisted_invalid_role"])
def test_session_creation_revalidates_trusted_identity(reset_db, identity, mutation):
    principal = authenticate(company_id=identity[2])
    with Session(reset_db) as session:
        if mutation == "inactive_user":
            session.get(User, identity[0]).is_active = False
        elif mutation == "inactive_membership":
            session.get(CompanyMembership, identity[1]).is_active = False
        elif mutation == "persisted_invalid_role":
            session.execute(sa.text(
                "ALTER TABLE company_memberships DROP CONSTRAINT ck_company_memberships_role_allowed"
            ))
            session.get(CompanyMembership, identity[1]).role = "corrupted-role"
        session.commit()
    if mutation == "company":
        principal = replace(principal, company_id=identity[2] + 1)
    elif mutation == "role":
        principal = replace(principal, role=MembershipRole.COMPANY_ADMIN)
    elif mutation == "user":
        principal = replace(principal, user_id=identity[0] + 1)
    elif mutation == "invalid_role":
        principal = replace(principal, role="not-a-role")
    elif mutation == "invalid_version":
        principal = replace(principal, password_version=True)
    before = snapshot(reset_db)
    with pytest.raises(AuthInfrastructureError, match="^Human authentication unavailable$"):
        humans.create_session(principal=principal)
    assert snapshot(reset_db) == before


def test_explicit_stale_principal_inserts_zero_sessions(reset_db, identity):
    principal = authenticate(company_id=identity[2])
    created = request_password_reset(email=EMAIL)
    reset_password(token=created.token, new_password=NEW_PASSWORD)
    before = snapshot(reset_db)
    assert before[2] == []
    with pytest.raises(StaleCredentialsError, match="^Human authentication unavailable$"):
        humans.create_session(principal=principal)
    assert snapshot(reset_db) == before


def test_reset_reuses_nonblank_policy_without_stripping_password(reset_db, identity):
    created = request_password_reset(email=EMAIL)
    reset_password(token=created.token, new_password=" x ")
    with Session(reset_db) as session:
        user = session.get(User, identity[0])
        assert verify_password(" x ", user.password_hash)
        assert not verify_password("x", user.password_hash)


@pytest.mark.parametrize("mutation", ["version", "routing_identity", "timestamp"])
def test_corrupted_trusted_reset_state_is_sanitized(monkeypatch, reset_db, identity, mutation):
    created = request_password_reset(email=EMAIL)
    if mutation == "version":
        with Session(reset_db) as session:
            session.execute(sa.text("ALTER TABLE users DROP CONSTRAINT ck_users_password_version_positive"))
            session.get(User, identity[0]).password_version = 0
            session.commit()
    else:
        real_lookup = resets.find_reset_token_by_digest

        def corrupted_lookup(session, **kwargs):
            row = real_lookup(session, **kwargs)
            if kwargs.get("for_update"):
                if mutation == "routing_identity":
                    row.user_id = identity[0] + 1
                else:
                    row.expires_at = None
            return row

        monkeypatch.setattr(resets, "find_reset_token_by_digest", corrupted_lookup)
    before = snapshot(reset_db)
    with pytest.raises(PasswordResetInfrastructureError, match="^Password reset unavailable$"):
        reset_password(token=created.token, new_password=NEW_PASSWORD)
    assert snapshot(reset_db) == before


@pytest.mark.parametrize("operation", ["request", "confirm"])
def test_lookup_database_failure_is_sanitized(monkeypatch, reset_db, identity, operation, caplog):
    created = request_password_reset(email=EMAIL)
    before = snapshot(reset_db)

    def fail(*args, **kwargs):
        raise RuntimeError(f"postgresql://private {created.token} {EMAIL} {OLD_HASH} {NEW_PASSWORD}")

    lookup = "find_user_by_normalized_email_for_update" if operation == "request" else "find_reset_token_by_digest"
    monkeypatch.setattr(resets, lookup, fail)
    with pytest.raises(PasswordResetInfrastructureError) as error:
        if operation == "request":
            request_password_reset(email=EMAIL)
        else:
            reset_password(token=created.token, new_password=NEW_PASSWORD)
    assert str(error.value) == "Password reset unavailable"
    assert error.value.__suppress_context__
    for secret in (created.token, EMAIL, OLD_HASH, NEW_PASSWORD):
        assert secret not in str(error.value)
        assert secret not in caplog.text
    assert snapshot(reset_db) == before


@pytest.mark.parametrize("stage", ["entropy", "digest", "commit"])
def test_request_generation_or_commit_failure_rolls_back_rotation(monkeypatch, reset_db, identity, stage):
    request_password_reset(email=EMAIL)
    before = snapshot(reset_db)

    def fail(*args, **kwargs):
        raise RuntimeError("private token generation or database failure")

    if stage == "entropy":
        monkeypatch.setattr(resets.secrets, "token_urlsafe", fail)
    elif stage == "digest":
        monkeypatch.setattr(resets, "_digest_token", fail)
    else:
        class FailingSession(Session):
            pass

        sa.event.listen(FailingSession, "before_commit", fail)
        monkeypatch.setattr(resets, "SessionLocal", sessionmaker(bind=reset_db, class_=FailingSession))
    with pytest.raises(PasswordResetInfrastructureError, match="^Password reset unavailable$"):
        request_password_reset(email=EMAIL)
    assert snapshot(reset_db) == before


def test_malformed_token_never_opens_transaction(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Malformed token opened a transaction")

    monkeypatch.setattr(resets, "SessionLocal", fail)
    with pytest.raises(InvalidPasswordResetTokenError, match="^Invalid password reset token$"):
        reset_password(token="not-a-token", new_password=NEW_PASSWORD)


def run_serialized_pair(monkeypatch, database, first, second):
    """Prove the second backend actually waits on the first User row lock."""
    acquired = threading.Event()
    attempted = threading.Event()
    release = threading.Event()
    pids, results, errors = {}, {}, {}

    def wrap(real):
        def coordinated(session, **kwargs):
            name = threading.current_thread().name
            pids[name] = session.scalar(sa.text("SELECT pg_backend_pid()"))
            if name == "second":
                attempted.set()
            result = real(session, **kwargs)
            if name == "first":
                acquired.set()
                assert release.wait(timeout=10)
            return result
        return coordinated

    for module, name in (
        (resets, "find_user_by_normalized_email_for_update"),
        (resets, "find_user_by_id_for_update"),
        (humans, "find_user_by_id_for_update"),
    ):
        monkeypatch.setattr(module, name, wrap(getattr(module, name)))

    def invoke(name, function):
        try:
            results[name] = function()
        except Exception as error:
            errors[name] = error

    threads = [threading.Thread(target=invoke, args=(name, function), name=name)
               for name, function in (("first", first), ("second", second))]
    try:
        threads[0].start()
        assert acquired.wait(timeout=5)
        threads[1].start()
        assert attempted.wait(timeout=5)
        deadline = time.monotonic() + 5
        with database.connect() as connection:
            while time.monotonic() < deadline:
                blockers = connection.scalar(sa.text("SELECT pg_blocking_pids(:pid)"), {"pid": pids["second"]})
                if pids["first"] in blockers:
                    break
                threading.Event().wait(0.01)
            else:
                pytest.fail("Second operation did not wait on the shared User row lock")
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=20)
    assert all(not thread.is_alive() for thread in threads)
    return results, errors


def test_postgresql_concurrent_requests_leave_one_open_token(monkeypatch, reset_db, identity):
    results, errors = run_serialized_pair(
        monkeypatch, reset_db,
        lambda: request_password_reset(email=EMAIL),
        lambda: request_password_reset(email=" RESET@EXAMPLE.COM "),
    )
    assert errors == {}
    assert len(results) == 2
    with Session(reset_db) as session:
        rows = list(session.scalars(sa.select(PasswordResetToken).order_by(PasswordResetToken.id)))
        assert len(rows) == 2
        assert rows[0].revoked_at is not None
        assert rows[1].revoked_at is rows[1].consumed_at is None
        assert rows[1].token_digest == hashlib.sha256(results["second"].token.encode("ascii")).digest()


def test_postgresql_concurrent_confirmation_exactly_one_success(monkeypatch, reset_db, identity):
    created = request_password_reset(email=EMAIL)
    results, errors = run_serialized_pair(
        monkeypatch, reset_db,
        lambda: reset_password(token=created.token, new_password=NEW_PASSWORD),
        lambda: reset_password(token=created.token, new_password="losing password"),
    )
    assert list(results) == ["first"]
    assert isinstance(errors.get("second"), InvalidPasswordResetTokenError)
    with Session(reset_db) as session:
        user = session.get(User, identity[0])
        assert user.password_version == 2
        assert verify_password(NEW_PASSWORD, user.password_hash)


@pytest.mark.parametrize("first_operation", ["request", "confirm"])
def test_postgresql_request_vs_confirmation_both_orders(monkeypatch, reset_db, identity, first_operation):
    created = request_password_reset(email=EMAIL)
    operations = {
        "request": lambda: request_password_reset(email=EMAIL),
        "confirm": lambda: reset_password(token=created.token, new_password=NEW_PASSWORD),
    }
    second_operation = "confirm" if first_operation == "request" else "request"
    results, errors = run_serialized_pair(
        monkeypatch, reset_db, operations[first_operation], operations[second_operation],
    )
    with Session(reset_db) as session:
        user = session.get(User, identity[0])
        rows = list(session.scalars(sa.select(PasswordResetToken).order_by(PasswordResetToken.id)))
        assert len(rows) == 2
        assert rows[1].consumed_at is rows[1].revoked_at is None
        if first_operation == "request":
            assert list(results) == ["first"]
            assert isinstance(errors.get("second"), InvalidPasswordResetTokenError)
            assert user.password_version == 1 and user.password_hash == OLD_HASH
            assert rows[0].revoked_at is not None and rows[0].consumed_at is None
        else:
            assert errors == {} and len(results) == 2
            assert user.password_version == 2 and verify_password(NEW_PASSWORD, user.password_hash)
            assert rows[0].consumed_at is not None and rows[0].revoked_at is None


@pytest.mark.parametrize("first_operation", ["session", "reset"])
def test_postgresql_session_vs_reset_no_old_credential_session_survives(monkeypatch, reset_db, identity, first_operation):
    principal = authenticate(company_id=identity[2])
    created = request_password_reset(email=EMAIL)
    operations = {
        "session": lambda: humans.create_session(principal=principal),
        "reset": lambda: reset_password(token=created.token, new_password=NEW_PASSWORD),
    }
    second_operation = "reset" if first_operation == "session" else "session"
    results, errors = run_serialized_pair(
        monkeypatch, reset_db, operations[first_operation], operations[second_operation],
    )
    with Session(reset_db) as session:
        assert session.get(User, identity[0]).password_version == 2
        sessions = list(session.scalars(sa.select(UserSession)))
        assert all(row.revoked_at is not None for row in sessions)
        if first_operation == "session":
            assert errors == {} and len(results) == 2 and len(sessions) == 1
        else:
            assert list(results) == ["first"] and sessions == []
            assert isinstance(errors.get("second"), StaleCredentialsError)
