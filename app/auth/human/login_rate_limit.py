"""Atomic backend primitives for human-login rate limiting."""

import hashlib
import ipaddress
import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from app.auth.human.exceptions import (
    LoginRateLimitExceededError,
    LoginRateLimitInfrastructureError,
)
from app.auth.human.types import LoginRateLimitPermit
from app.config import settings
from app.domain.users import normalize_email

IP_ATTEMPT_LIMIT = 10
IP_ATTEMPT_WINDOW_SECONDS = 5 * 60
EMAIL_FAILURE_LIMIT = 10
EMAIL_FAILURE_WINDOW_SECONDS = 15 * 60
PAIR_FAILURE_LIMIT = 5
PAIR_FAILURE_WINDOW_SECONDS = 5 * 60
LOGIN_RATE_LIMIT_PERMIT_LIFETIME_SECONDS = 5 * 60

_INFRASTRUCTURE_MESSAGE = "Login rate limiting unavailable"


class LoginRateLimitClock(Protocol):
    def now(self) -> float: ...


class LoginRateLimitStore(Protocol):
    def admit_attempt(
        self,
        *,
        attempt_id: str,
        ip_key: bytes,
        email_key: bytes | None,
        pair_key: bytes | None,
    ) -> int | None: ...

    def record_failure(self, *, attempt_id: str) -> None: ...

    def record_success(self, *, attempt_id: str) -> None: ...

    def record_aborted(self, *, attempt_id: str) -> None: ...


class MonotonicLoginRateLimitClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class _PermitState:
    ip_key: bytes
    email_key: bytes | None
    pair_key: bytes | None
    expires_at: float


class _InvalidPermitError(Exception):
    pass


class InMemoryLoginRateLimitStore:
    """Process-local atomic store intended only for local and test use."""

    def __init__(self, *, clock: LoginRateLimitClock | None = None):
        self._lock = threading.RLock()
        self._clock = clock or MonotonicLoginRateLimitClock()
        self._ip_attempts: dict[bytes, deque[float]] = {}
        self._email_failures: dict[bytes, deque[float]] = {}
        self._pair_failures: dict[bytes, deque[float]] = {}
        self._permits: dict[str, _PermitState] = {}

    def admit_attempt(
        self,
        *,
        attempt_id: str,
        ip_key: bytes,
        email_key: bytes | None,
        pair_key: bytes | None,
    ) -> int | None:
        with self._lock:
            now = self._clock.now()
            self._cleanup(now=now)
            if attempt_id in self._permits:
                raise RuntimeError("rate-limit attempt identifier collision")

            email_pending_expirations = (
                [
                    permit.expires_at
                    for permit in self._permits.values()
                    if permit.email_key == email_key
                ]
                if email_key is not None
                else []
            )
            pair_pending_expirations = (
                [
                    permit.expires_at
                    for permit in self._permits.values()
                    if permit.pair_key == pair_key
                ]
                if pair_key is not None
                else []
            )
            blocking_durations = [
                duration
                for duration in (
                    self._blocking_duration(
                        events=self._ip_attempts.get(ip_key),
                        limit=IP_ATTEMPT_LIMIT,
                        window_seconds=IP_ATTEMPT_WINDOW_SECONDS,
                        now=now,
                    ),
                    (
                        self._credential_blocking_duration(
                            events=self._email_failures.get(email_key),
                            pending_expirations=email_pending_expirations,
                            limit=EMAIL_FAILURE_LIMIT,
                            window_seconds=EMAIL_FAILURE_WINDOW_SECONDS,
                            now=now,
                        )
                        if email_key is not None
                        else None
                    ),
                    (
                        self._credential_blocking_duration(
                            events=self._pair_failures.get(pair_key),
                            pending_expirations=pair_pending_expirations,
                            limit=PAIR_FAILURE_LIMIT,
                            window_seconds=PAIR_FAILURE_WINDOW_SECONDS,
                            now=now,
                        )
                        if pair_key is not None
                        else None
                    ),
                )
                if duration is not None
            ]
            if blocking_durations:
                return max(1, math.ceil(max(blocking_durations)))

            self._events_for(self._ip_attempts, ip_key).append(now)
            self._permits[attempt_id] = _PermitState(
                ip_key=ip_key,
                email_key=email_key,
                pair_key=pair_key,
                expires_at=(
                    now + LOGIN_RATE_LIMIT_PERMIT_LIFETIME_SECONDS
                ),
            )
            return None

    def record_failure(self, *, attempt_id: str) -> None:
        with self._lock:
            now = self._clock.now()
            self._cleanup(now=now)
            permit = self._consume_permit(attempt_id=attempt_id)
            if permit.email_key is not None:
                self._events_for(
                    self._email_failures,
                    permit.email_key,
                ).append(now)
            if permit.pair_key is not None:
                self._events_for(
                    self._pair_failures,
                    permit.pair_key,
                ).append(now)

    def record_success(self, *, attempt_id: str) -> None:
        with self._lock:
            now = self._clock.now()
            self._cleanup(now=now)
            self._consume_permit(attempt_id=attempt_id)

    def record_aborted(self, *, attempt_id: str) -> None:
        with self._lock:
            now = self._clock.now()
            self._cleanup(now=now)
            self._consume_permit(attempt_id=attempt_id)

    def _state_counts_for_testing(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                sum(len(events) for events in self._ip_attempts.values()),
                sum(len(events) for events in self._email_failures.values()),
                sum(len(events) for events in self._pair_failures.values()),
                len(self._permits),
            )

    def _reservation_counts_for_testing(self) -> tuple[int, int]:
        with self._lock:
            return (
                sum(
                    permit.email_key is not None
                    for permit in self._permits.values()
                ),
                sum(
                    permit.pair_key is not None
                    for permit in self._permits.values()
                ),
            )

    def _consume_permit(self, *, attempt_id: str) -> _PermitState:
        try:
            return self._permits.pop(attempt_id)
        except KeyError:
            raise _InvalidPermitError from None

    def _cleanup(self, *, now: float) -> None:
        self._cleanup_events(
            histories=self._ip_attempts,
            cutoff=now - IP_ATTEMPT_WINDOW_SECONDS,
        )
        self._cleanup_events(
            histories=self._email_failures,
            cutoff=now - EMAIL_FAILURE_WINDOW_SECONDS,
        )
        self._cleanup_events(
            histories=self._pair_failures,
            cutoff=now - PAIR_FAILURE_WINDOW_SECONDS,
        )
        expired_attempt_ids = [
            attempt_id
            for attempt_id, permit in self._permits.items()
            if permit.expires_at <= now
        ]
        for attempt_id in expired_attempt_ids:
            del self._permits[attempt_id]

    @staticmethod
    def _cleanup_events(
        *,
        histories: dict[bytes, deque[float]],
        cutoff: float,
    ) -> None:
        empty_keys = []
        for key, events in histories.items():
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                empty_keys.append(key)
        for key in empty_keys:
            del histories[key]

    @staticmethod
    def _blocking_duration(
        *,
        events: deque[float] | None,
        limit: int,
        window_seconds: int,
        now: float,
    ) -> float | None:
        if events is None or len(events) < limit:
            return None
        event_that_must_expire = events[len(events) - limit]
        return window_seconds - (now - event_that_must_expire)

    @staticmethod
    def _credential_blocking_duration(
        *,
        events: deque[float] | None,
        pending_expirations: list[float],
        limit: int,
        window_seconds: int,
        now: float,
    ) -> float | None:
        expiration_times = [
            event + window_seconds
            for event in events or ()
        ]
        expiration_times.extend(pending_expirations)
        if len(expiration_times) < limit:
            return None
        expiration_times.sort()
        expiration_that_must_pass = expiration_times[
            len(expiration_times) - limit
        ]
        return expiration_that_must_pass - now

    @staticmethod
    def _events_for(
        histories: dict[bytes, deque[float]],
        key: bytes,
    ) -> deque[float]:
        events = histories.get(key)
        if events is None:
            events = deque()
            histories[key] = events
        return events


class LoginRateLimiter:
    def __init__(
        self,
        *,
        store: LoginRateLimitStore,
    ):
        self._store = store

    def check(self, *, client_ip: str, email: str) -> LoginRateLimitPermit:
        try:
            ip_key, email_key, pair_key = _derive_limiter_keys(
                client_ip=client_ip,
                email=email,
            )
            attempt_id = secrets.token_urlsafe(32)
            retry_after_seconds = self._store.admit_attempt(
                attempt_id=attempt_id,
                ip_key=ip_key,
                email_key=email_key,
                pair_key=pair_key,
            )
        except LoginRateLimitInfrastructureError:
            raise
        except Exception:
            raise LoginRateLimitInfrastructureError(
                _INFRASTRUCTURE_MESSAGE
            ) from None
        if retry_after_seconds is not None:
            if (
                not isinstance(retry_after_seconds, int)
                or isinstance(retry_after_seconds, bool)
                or retry_after_seconds < 1
            ):
                raise LoginRateLimitInfrastructureError(
                    _INFRASTRUCTURE_MESSAGE
                )
            raise LoginRateLimitExceededError(retry_after_seconds)
        return LoginRateLimitPermit(attempt_id=attempt_id)

    def record_failure(self, *, permit: LoginRateLimitPermit) -> None:
        try:
            attempt_id = _validate_permit(permit=permit)
            self._store.record_failure(attempt_id=attempt_id)
        except LoginRateLimitInfrastructureError:
            raise
        except Exception:
            raise LoginRateLimitInfrastructureError(
                _INFRASTRUCTURE_MESSAGE
            ) from None

    def record_success(self, *, permit: LoginRateLimitPermit) -> None:
        try:
            attempt_id = _validate_permit(permit=permit)
            self._store.record_success(attempt_id=attempt_id)
        except LoginRateLimitInfrastructureError:
            raise
        except Exception:
            raise LoginRateLimitInfrastructureError(
                _INFRASTRUCTURE_MESSAGE
            ) from None

    def record_aborted(self, *, permit: LoginRateLimitPermit) -> None:
        try:
            attempt_id = _validate_permit(permit=permit)
            self._store.record_aborted(attempt_id=attempt_id)
        except LoginRateLimitInfrastructureError:
            raise
        except Exception:
            raise LoginRateLimitInfrastructureError(
                _INFRASTRUCTURE_MESSAGE
            ) from None


class _DefaultLoginRateLimiterProvider:
    def __init__(self):
        self._lock = threading.Lock()
        self._local_limiter: LoginRateLimiter | None = None
        self._redis_limiter: LoginRateLimiter | None = None

    def get(self) -> LoginRateLimiter:
        app_env = settings.APP_ENV
        backend = settings.LOGIN_RATE_LIMIT_BACKEND
        if backend == "redis":
            with self._lock:
                if self._redis_limiter is None:
                    try:
                        from app.infrastructure.redis import create_redis_client
                        from app.auth.human.redis_login_rate_limit_store import (
                            RedisLoginRateLimitStore,
                        )

                        self._redis_limiter = LoginRateLimiter(
                            store=RedisLoginRateLimitStore(
                                client=create_redis_client(settings),
                            ),
                        )
                    except Exception:
                        raise LoginRateLimitInfrastructureError(
                            _INFRASTRUCTURE_MESSAGE
                        ) from None
                return self._redis_limiter
        if app_env not in {"local", "test"} or backend not in {None, "memory"}:
            raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE)

        with self._lock:
            if self._local_limiter is None:
                self._local_limiter = LoginRateLimiter(
                    store=InMemoryLoginRateLimitStore(),
                )
            return self._local_limiter


_default_provider = _DefaultLoginRateLimiterProvider()


def check_login_rate_limit(
    *,
    client_ip: str,
    email: str,
) -> LoginRateLimitPermit:
    try:
        return _default_provider.get().check(client_ip=client_ip, email=email)
    except (LoginRateLimitExceededError, LoginRateLimitInfrastructureError):
        raise
    except Exception:
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None


def record_login_failure(
    *,
    permit: LoginRateLimitPermit,
) -> None:
    try:
        _default_provider.get().record_failure(permit=permit)
    except LoginRateLimitInfrastructureError:
        raise
    except Exception:
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None


def record_login_success(
    *,
    permit: LoginRateLimitPermit,
) -> None:
    try:
        _default_provider.get().record_success(permit=permit)
    except LoginRateLimitInfrastructureError:
        raise
    except Exception:
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None


def record_login_aborted(
    *,
    permit: LoginRateLimitPermit,
) -> None:
    try:
        _default_provider.get().record_aborted(permit=permit)
    except LoginRateLimitInfrastructureError:
        raise
    except Exception:
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None


def _derive_limiter_keys(
    *,
    client_ip: str,
    email: str,
) -> tuple[bytes, bytes | None, bytes | None]:
    try:
        if not isinstance(client_ip, str) or not client_ip.strip():
            raise ValueError
        canonical_ip = ipaddress.ip_address(client_ip.strip()).compressed
    except (AttributeError, TypeError, ValueError):
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None
    except Exception:
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None

    ip_key = _digest_key(namespace="ip", value=canonical_ip)
    try:
        normalized_email = normalize_email(email)
    except (AttributeError, TypeError, ValueError):
        return ip_key, None, None
    except Exception:
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE) from None

    email_key = _digest_key(namespace="email", value=normalized_email)
    pair_key = _digest_key(
        namespace="pair",
        value=f"{canonical_ip}\0{normalized_email}",
    )
    return ip_key, email_key, pair_key


def _digest_key(*, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).digest()


def _validate_permit(*, permit: LoginRateLimitPermit) -> str:
    if (
        not isinstance(permit, LoginRateLimitPermit)
        or not isinstance(permit.attempt_id, str)
        or not permit.attempt_id
    ):
        raise LoginRateLimitInfrastructureError(_INFRASTRUCTURE_MESSAGE)
    return permit.attempt_id
