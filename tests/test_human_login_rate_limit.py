import threading
from dataclasses import fields

import pytest

from app.auth.human import (
    LoginRateLimitExceededError,
    LoginRateLimitInfrastructureError,
    LoginRateLimitPermit,
)
from app.auth.human import login_rate_limit as rate_limit
from app.domain.users import normalize_email as canonical_normalize_email


class FakeClock:
    def __init__(self, now=0.0):
        self.current = now

    def now(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds


def _new_limiter():
    clock = FakeClock()
    store = rate_limit.InMemoryLoginRateLimitStore(clock=clock)
    limiter = rate_limit.LoginRateLimiter(store=store)
    return limiter, store, clock


def _record_failure(limiter, *, client_ip, email):
    permit = limiter.check(client_ip=client_ip, email=email)
    limiter.record_failure(permit=permit)
    return permit


def _assert_exceeded(limiter, *, client_ip, email, retry_after_seconds):
    with pytest.raises(LoginRateLimitExceededError) as exc_info:
        limiter.check(client_ip=client_ip, email=email)
    assert str(exc_info.value) == "Login rate limit exceeded"
    assert exc_info.value.retry_after_seconds == retry_after_seconds
    return exc_info.value


def test_ip_attempts_one_through_ten_are_admitted_then_next_is_blocked():
    limiter, store, _ = _new_limiter()

    permits = [
        limiter.check(client_ip="192.0.2.10", email=f"user-{index}@example.com")
        for index in range(10)
    ]

    assert all(isinstance(permit, LoginRateLimitPermit) for permit in permits)
    _assert_exceeded(
        limiter,
        client_ip="192.0.2.10",
        email="next@example.com",
        retry_after_seconds=300,
    )
    assert store._state_counts_for_testing() == (10, 0, 0, 10)


def test_ip_attempt_ages_out_at_exact_five_minute_boundary():
    limiter, _, clock = _new_limiter()
    for index in range(10):
        limiter.check(
            client_ip="192.0.2.11",
            email=f"boundary-{index}@example.com",
        )

    clock.advance(299.001)
    _assert_exceeded(
        limiter,
        client_ip="192.0.2.11",
        email="before@example.com",
        retry_after_seconds=1,
    )
    clock.advance(0.999)

    assert isinstance(
        limiter.check(
            client_ip="192.0.2.11",
            email="at-boundary@example.com",
        ),
        LoginRateLimitPermit,
    )


def test_email_failures_one_through_ten_then_block_precheck():
    limiter, store, _ = _new_limiter()
    for index in range(10):
        _record_failure(
            limiter,
            client_ip=f"198.51.100.{index + 1}",
            email="victim@example.com",
        )

    _assert_exceeded(
        limiter,
        client_ip="198.51.100.200",
        email="victim@example.com",
        retry_after_seconds=900,
    )
    assert store._state_counts_for_testing() == (10, 10, 10, 0)


def test_email_failure_ages_out_at_exact_fifteen_minute_boundary():
    limiter, _, clock = _new_limiter()
    for index in range(10):
        _record_failure(
            limiter,
            client_ip=f"203.0.113.{index + 1}",
            email="email-boundary@example.com",
        )

    clock.advance(899.5)
    _assert_exceeded(
        limiter,
        client_ip="203.0.113.200",
        email="email-boundary@example.com",
        retry_after_seconds=1,
    )
    clock.advance(0.5)

    assert isinstance(
        limiter.check(
            client_ip="203.0.113.200",
            email="email-boundary@example.com",
        ),
        LoginRateLimitPermit,
    )


def test_pair_failures_one_through_five_then_block_precheck():
    limiter, store, _ = _new_limiter()
    for _ in range(5):
        _record_failure(
            limiter,
            client_ip="192.0.2.20",
            email="pair@example.com",
        )

    _assert_exceeded(
        limiter,
        client_ip="192.0.2.20",
        email="pair@example.com",
        retry_after_seconds=300,
    )
    assert store._state_counts_for_testing() == (5, 5, 5, 0)


def test_pair_failure_ages_out_at_exact_five_minute_boundary():
    limiter, _, clock = _new_limiter()
    for _ in range(5):
        _record_failure(
            limiter,
            client_ip="192.0.2.21",
            email="pair-boundary@example.com",
        )

    clock.advance(299.25)
    _assert_exceeded(
        limiter,
        client_ip="192.0.2.21",
        email="pair-boundary@example.com",
        retry_after_seconds=1,
    )
    clock.advance(0.75)

    assert isinstance(
        limiter.check(
            client_ip="192.0.2.21",
            email="pair-boundary@example.com",
        ),
        LoginRateLimitPermit,
    )


def test_ip_limit_counts_successful_and_failed_admitted_attempts():
    limiter, _, _ = _new_limiter()
    for index in range(10):
        permit = limiter.check(
            client_ip="192.0.2.30",
            email=f"outcome-{index}@example.com",
        )
        if index % 2:
            limiter.record_failure(permit=permit)
        else:
            limiter.record_success(permit=permit)

    _assert_exceeded(
        limiter,
        client_ip="192.0.2.30",
        email="blocked@example.com",
        retry_after_seconds=300,
    )


@pytest.mark.parametrize("terminal_outcome", ["success", "aborted"])
def test_nonfailure_outcome_does_not_clear_email_failure_history(
    terminal_outcome,
):
    limiter, store, _ = _new_limiter()
    for index in range(9):
        _record_failure(
            limiter,
            client_ip=f"198.51.100.{index + 1}",
            email="persistent-email@example.com",
        )
    permit = limiter.check(
        client_ip="198.51.100.10",
        email="persistent-email@example.com",
    )
    getattr(limiter, f"record_{terminal_outcome}")(permit=permit)
    assert store._reservation_counts_for_testing() == (0, 0)
    _record_failure(
        limiter,
        client_ip="198.51.100.11",
        email="persistent-email@example.com",
    )

    _assert_exceeded(
        limiter,
        client_ip="198.51.100.200",
        email="persistent-email@example.com",
        retry_after_seconds=900,
    )


@pytest.mark.parametrize("terminal_outcome", ["success", "aborted"])
def test_nonfailure_outcome_does_not_clear_pair_failure_history(
    terminal_outcome,
):
    limiter, store, _ = _new_limiter()
    for _ in range(4):
        _record_failure(
            limiter,
            client_ip="192.0.2.40",
            email="persistent-pair@example.com",
        )
    permit = limiter.check(
        client_ip="192.0.2.40",
        email="persistent-pair@example.com",
    )
    getattr(limiter, f"record_{terminal_outcome}")(permit=permit)
    assert store._reservation_counts_for_testing() == (0, 0)
    _record_failure(
        limiter,
        client_ip="192.0.2.40",
        email="persistent-pair@example.com",
    )

    _assert_exceeded(
        limiter,
        client_ip="192.0.2.40",
        email="persistent-pair@example.com",
        retry_after_seconds=300,
    )


def test_unknown_and_wrong_password_flows_have_identical_limiter_behavior():
    unknown_limiter, _, _ = _new_limiter()
    known_limiter, _, _ = _new_limiter()
    for _ in range(5):
        _record_failure(
            unknown_limiter,
            client_ip="192.0.2.50",
            email="unknown@example.com",
        )
        _record_failure(
            known_limiter,
            client_ip="192.0.2.51",
            email="known@example.com",
        )

    unknown_error = _assert_exceeded(
        unknown_limiter,
        client_ip="192.0.2.50",
        email="unknown@example.com",
        retry_after_seconds=300,
    )
    known_error = _assert_exceeded(
        known_limiter,
        client_ip="192.0.2.51",
        email="known@example.com",
        retry_after_seconds=300,
    )
    assert type(unknown_error) is type(known_error)
    assert unknown_error.args == known_error.args
    assert unknown_error.__dict__ == known_error.__dict__


def test_canonical_normalize_email_is_reused(monkeypatch):
    limiter, _, _ = _new_limiter()
    normalized_inputs = []

    def recording_normalize_email(email):
        normalized_inputs.append(email)
        return canonical_normalize_email(email)

    monkeypatch.setattr(rate_limit, "normalize_email", recording_normalize_email)
    for index in range(10):
        attempted_email = (
            " User@Example.COM " if index % 2 else "user@example.com"
        )
        _record_failure(
            limiter,
            client_ip=f"203.0.113.{index + 1}",
            email=attempted_email,
        )

    _assert_exceeded(
        limiter,
        client_ip="203.0.113.200",
        email="USER@example.com",
        retry_after_seconds=900,
    )
    assert len(normalized_inputs) == 11


def test_multiple_blockers_use_ceiling_of_maximum_remaining_duration():
    limiter, _, clock = _new_limiter()
    client_ip = "192.0.2.60"
    email = "multiple@example.com"
    for _ in range(5):
        _record_failure(limiter, client_ip=client_ip, email=email)

    clock.advance(100)
    for index in range(5):
        _record_failure(
            limiter,
            client_ip=f"198.51.100.{index + 1}",
            email=email,
        )
    for index in range(5):
        permit = limiter.check(
            client_ip=client_ip,
            email=f"ip-fill-{index}@example.com",
        )
        limiter.record_success(permit=permit)

    error = _assert_exceeded(
        limiter,
        client_ip=client_ip,
        email=email,
        retry_after_seconds=800,
    )
    assert error.__dict__ == {"retry_after_seconds": 800}
    assert "email" not in str(error).lower()
    assert "pair" not in str(error).lower()
    assert "ip" not in str(error).lower()

    clock.advance(799.5)
    _assert_exceeded(
        limiter,
        client_ip="203.0.113.250",
        email=email,
        retry_after_seconds=1,
    )


def test_precheck_returns_opaque_permit_without_incrementing_failures():
    limiter, store, _ = _new_limiter()

    first = limiter.check(client_ip="192.0.2.70", email="opaque@example.com")
    second = limiter.check(client_ip="192.0.2.71", email="opaque@example.com")

    assert [(field.name, field.type) for field in fields(first)] == [
        ("attempt_id", str)
    ]
    assert first.attempt_id != second.attempt_id
    assert len(first.attempt_id) >= 32
    assert "opaque@example.com" not in repr(first)
    assert "192.0.2.70" not in repr(first)
    assert store._state_counts_for_testing() == (2, 0, 0, 2)


@pytest.mark.parametrize(
    ("first_outcome", "second_outcome", "expected_failure_events"),
    [
        ("failure", "failure", 1),
        ("failure", "success", 1),
        ("failure", "aborted", 1),
        ("success", "failure", 0),
        ("success", "success", 0),
        ("success", "aborted", 0),
        ("aborted", "failure", 0),
        ("aborted", "success", 0),
        ("aborted", "aborted", 0),
    ],
)
def test_permit_accepts_exactly_one_terminal_outcome(
    first_outcome,
    second_outcome,
    expected_failure_events,
):
    limiter, store, _ = _new_limiter()
    permit = limiter.check(client_ip="192.0.2.80", email="once@example.com")

    getattr(limiter, f"record_{first_outcome}")(permit=permit)
    with pytest.raises(LoginRateLimitInfrastructureError) as exc_info:
        getattr(limiter, f"record_{second_outcome}")(permit=permit)

    assert str(exc_info.value) == "Login rate limiting unavailable"
    assert exc_info.value.__cause__ is None
    assert store._state_counts_for_testing() == (
        1,
        expected_failure_events,
        expected_failure_events,
        0,
    )
    assert store._reservation_counts_for_testing() == (0, 0)


def test_abandoned_permit_expires_at_exact_lifetime_boundary():
    limiter, store, clock = _new_limiter()
    permit = limiter.check(client_ip="192.0.2.90", email="abandoned@example.com")

    clock.advance(rate_limit.LOGIN_RATE_LIMIT_PERMIT_LIFETIME_SECONDS)
    with pytest.raises(LoginRateLimitInfrastructureError):
        limiter.record_failure(permit=permit)

    assert store._state_counts_for_testing() == (0, 0, 0, 0)


def test_permit_remains_valid_immediately_before_lifetime_boundary():
    limiter, store, clock = _new_limiter()
    permit = limiter.check(
        client_ip="192.0.2.91",
        email="permit-boundary@example.com",
    )

    clock.advance(rate_limit.LOGIN_RATE_LIMIT_PERMIT_LIFETIME_SECONDS - 0.001)

    assert limiter.record_success(permit=permit) is None
    assert store._state_counts_for_testing() == (1, 0, 0, 0)


def test_cleanup_prevents_abandoned_permits_from_accumulating():
    limiter, store, clock = _new_limiter()
    for index in range(10):
        limiter.check(
            client_ip=f"198.51.100.{index + 1}",
            email=f"abandoned-{index}@example.com",
        )
    assert store._state_counts_for_testing() == (10, 0, 0, 10)
    assert store._reservation_counts_for_testing() == (10, 10)

    clock.advance(rate_limit.LOGIN_RATE_LIMIT_PERMIT_LIFETIME_SECONDS)
    limiter.check(client_ip="198.51.100.200", email="current@example.com")

    assert store._state_counts_for_testing() == (1, 0, 0, 1)
    assert store._reservation_counts_for_testing() == (1, 1)


def test_in_memory_store_instances_are_isolated():
    first_limiter, first_store, _ = _new_limiter()
    second_limiter, second_store, _ = _new_limiter()
    first_limiter.check(client_ip="192.0.2.100", email="isolated@example.com")

    assert first_store._state_counts_for_testing() == (1, 0, 0, 1)
    assert second_store._state_counts_for_testing() == (0, 0, 0, 0)
    assert isinstance(
        second_limiter.check(
            client_ip="192.0.2.100",
            email="isolated@example.com",
        ),
        LoginRateLimitPermit,
    )


def test_concurrent_prechecks_cannot_exceed_ip_boundary():
    limiter, store, _ = _new_limiter()
    barrier = threading.Barrier(20)
    permits = []
    errors = []
    result_lock = threading.Lock()

    def check(index):
        barrier.wait(timeout=5)
        try:
            result = limiter.check(
                client_ip="192.0.2.110",
                email=f"concurrent-{index}@example.com",
            )
            with result_lock:
                permits.append(result)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=check, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(thread.is_alive() is False for thread in threads)
    assert len(permits) == 10
    assert len(errors) == 10
    assert all(isinstance(error, LoginRateLimitExceededError) for error in errors)
    assert store._state_counts_for_testing() == (10, 0, 0, 10)


def test_concurrent_prechecks_cannot_exceed_pair_reservation_boundary():
    limiter, store, _ = _new_limiter()
    barrier = threading.Barrier(12)
    permits = []
    errors = []
    result_lock = threading.Lock()

    def check():
        barrier.wait(timeout=5)
        try:
            result = limiter.check(
                client_ip="192.0.2.111",
                email="concurrent-pair-reservation@example.com",
            )
            with result_lock:
                permits.append(result)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=check) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(thread.is_alive() is False for thread in threads)
    assert len(permits) == rate_limit.PAIR_FAILURE_LIMIT
    assert len(errors) == 12 - rate_limit.PAIR_FAILURE_LIMIT
    assert all(isinstance(error, LoginRateLimitExceededError) for error in errors)
    assert all(error.retry_after_seconds == 300 for error in errors)
    assert store._state_counts_for_testing() == (5, 0, 0, 5)
    assert store._reservation_counts_for_testing() == (5, 5)


def test_concurrent_prechecks_cannot_exceed_email_reservation_boundary():
    limiter, store, _ = _new_limiter()
    barrier = threading.Barrier(15)
    permits = []
    errors = []
    result_lock = threading.Lock()

    def check(index):
        barrier.wait(timeout=5)
        try:
            result = limiter.check(
                client_ip=f"198.51.100.{index + 1}",
                email="concurrent-email-reservation@example.com",
            )
            with result_lock:
                permits.append(result)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=check, args=(index,))
        for index in range(15)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(thread.is_alive() is False for thread in threads)
    assert len(permits) == rate_limit.EMAIL_FAILURE_LIMIT
    assert len(errors) == 15 - rate_limit.EMAIL_FAILURE_LIMIT
    assert all(isinstance(error, LoginRateLimitExceededError) for error in errors)
    assert all(error.retry_after_seconds == 300 for error in errors)
    assert store._state_counts_for_testing() == (10, 0, 0, 10)
    assert store._reservation_counts_for_testing() == (10, 10)


def test_failure_atomically_converts_reservation_to_history():
    limiter, store, _ = _new_limiter()
    permit = limiter.check(
        client_ip="192.0.2.112",
        email="reservation-to-failure@example.com",
    )
    assert store._reservation_counts_for_testing() == (1, 1)

    limiter.record_failure(permit=permit)

    assert store._state_counts_for_testing() == (1, 1, 1, 0)
    assert store._reservation_counts_for_testing() == (0, 0)


def test_concurrent_duplicate_failures_count_permit_once():
    limiter, store, _ = _new_limiter()
    permit = limiter.check(client_ip="192.0.2.120", email="duplicate@example.com")
    barrier = threading.Barrier(10)
    successes = []
    errors = []
    result_lock = threading.Lock()

    def record():
        barrier.wait(timeout=5)
        try:
            limiter.record_failure(permit=permit)
            with result_lock:
                successes.append(None)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=record) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(thread.is_alive() is False for thread in threads)
    assert successes == [None]
    assert len(errors) == 9
    assert all(
        isinstance(error, LoginRateLimitInfrastructureError) for error in errors
    )
    assert store._state_counts_for_testing() == (1, 1, 1, 0)


def test_concurrent_distinct_failures_enforce_pair_limit():
    limiter, store, _ = _new_limiter()
    permits = [
        limiter.check(
            client_ip="192.0.2.121",
            email="concurrent-pair@example.com",
        )
        for _ in range(5)
    ]
    barrier = threading.Barrier(5)
    errors = []
    result_lock = threading.Lock()

    def record(permit):
        barrier.wait(timeout=5)
        try:
            limiter.record_failure(permit=permit)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=record, args=(permit,)) for permit in permits
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    _assert_exceeded(
        limiter,
        client_ip="192.0.2.121",
        email="concurrent-pair@example.com",
        retry_after_seconds=300,
    )
    assert store._state_counts_for_testing() == (5, 5, 5, 0)


class FailingStore:
    def admit_attempt(self, **kwargs):
        raise RuntimeError("redis://private limiter detail")

    def record_failure(self, **kwargs):
        raise RuntimeError("redis://private limiter detail")

    def record_success(self, **kwargs):
        raise RuntimeError("redis://private limiter detail")

    def record_aborted(self, **kwargs):
        raise RuntimeError("redis://private limiter detail")


@pytest.mark.parametrize("failure_source", ["entropy", "clock"])
def test_internal_dependency_failures_are_sanitized(monkeypatch, failure_source):
    secret_detail = "private entropy or clock detail"
    clock = FakeClock()
    store = rate_limit.InMemoryLoginRateLimitStore(clock=clock)
    limiter = rate_limit.LoginRateLimiter(store=store)

    def fail(*args, **kwargs):
        raise RuntimeError(secret_detail)

    if failure_source == "entropy":
        monkeypatch.setattr(rate_limit.secrets, "token_urlsafe", fail)
    else:
        monkeypatch.setattr(clock, "now", fail)

    with pytest.raises(LoginRateLimitInfrastructureError) as exc_info:
        limiter.check(client_ip="192.0.2.129", email="failure@example.com")

    assert str(exc_info.value) == "Login rate limiting unavailable"
    assert secret_detail not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert store._state_counts_for_testing() == (0, 0, 0, 0)


@pytest.mark.parametrize("operation", ["check", "failure", "success", "aborted"])
def test_storage_failures_are_sanitized_and_never_bypassed(operation):
    limiter = rate_limit.LoginRateLimiter(store=FailingStore())

    with pytest.raises(LoginRateLimitInfrastructureError) as exc_info:
        if operation == "check":
            limiter.check(client_ip="192.0.2.130", email="failure@example.com")
        elif operation == "failure":
            limiter.record_failure(permit=LoginRateLimitPermit("attempt"))
        elif operation == "success":
            limiter.record_success(permit=LoginRateLimitPermit("attempt"))
        else:
            limiter.record_aborted(permit=LoginRateLimitPermit("attempt"))

    assert str(exc_info.value) == "Login rate limiting unavailable"
    assert "redis" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("app_env", "backend"),
    [
        ("staging", None),
        ("production", None),
        ("staging", "memory"),
        ("production", "memory"),
        ("production", "redis"),
        ("local", "unsupported"),
    ],
)
def test_default_provider_fails_closed_for_unsafe_or_unsupported_storage(
    monkeypatch,
    app_env,
    backend,
):
    monkeypatch.setattr(rate_limit.settings, "APP_ENV", app_env)
    monkeypatch.setattr(rate_limit.settings, "LOGIN_RATE_LIMIT_BACKEND", backend)
    monkeypatch.setattr(
        rate_limit,
        "_default_provider",
        rate_limit._DefaultLoginRateLimiterProvider(),
    )

    with pytest.raises(LoginRateLimitInfrastructureError) as exc_info:
        rate_limit.check_login_rate_limit(
            client_ip="192.0.2.140",
            email="deployed@example.com",
        )

    assert str(exc_info.value) == "Login rate limiting unavailable"
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_default_provider_allows_memory_only_for_local_and_test(
    monkeypatch,
    app_env,
):
    monkeypatch.setattr(rate_limit.settings, "APP_ENV", app_env)
    monkeypatch.setattr(rate_limit.settings, "LOGIN_RATE_LIMIT_BACKEND", "memory")
    provider = rate_limit._DefaultLoginRateLimiterProvider()
    monkeypatch.setattr(rate_limit, "_default_provider", provider)

    permit = rate_limit.check_login_rate_limit(
        client_ip="192.0.2.150",
        email="local@example.com",
    )

    assert isinstance(permit, LoginRateLimitPermit)
    assert provider.get() is provider.get()


@pytest.mark.parametrize(
    "client_ip",
    [
        None,
        "",
        "not-an-ip",
    ],
)
def test_invalid_source_input_fails_closed_without_details(client_ip):
    limiter, _, _ = _new_limiter()

    with pytest.raises(LoginRateLimitInfrastructureError) as exc_info:
        limiter.check(client_ip=client_ip, email="user@example.com")

    assert str(exc_info.value) == "Login rate limiting unavailable"
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("email", [None, "", "   "])
@pytest.mark.parametrize("outcome", ["failure", "success", "aborted"])
def test_noncanonical_identity_uses_source_only_permit(email, outcome):
    limiter, store, _ = _new_limiter()

    permit = limiter.check(client_ip="192.0.2.160", email=email)
    assert store._reservation_counts_for_testing() == (0, 0)
    getattr(limiter, f"record_{outcome}")(permit=permit)

    assert store._state_counts_for_testing() == (1, 0, 0, 0)


def test_noncanonical_identity_attempts_still_enforce_source_limit():
    limiter, store, _ = _new_limiter()
    for _ in range(10):
        permit = limiter.check(client_ip="192.0.2.161", email="   ")
        limiter.record_failure(permit=permit)

    _assert_exceeded(
        limiter,
        client_ip="192.0.2.161",
        email=None,
        retry_after_seconds=300,
    )
    assert store._state_counts_for_testing() == (10, 0, 0, 0)
