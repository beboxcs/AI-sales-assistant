"""Real Redis tests; set B29_TEST_REDIS_URL to an isolated test server.

Each case owns a random namespace and deletes only that namespace. Window tests
freeze a Redis TIME snapshot in test-only script copies, avoiding 15-minute sleeps
and worker clocks. Production scripts always obtain Redis TIME themselves.
"""

import os
import secrets
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from redis import Redis
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

from app.auth.human import login_rate_limit as policy
from app.auth.human import redis_login_rate_limit_store as backend
from app.auth.human.exceptions import (
    LoginRateLimitExceededError, LoginRateLimitInfrastructureError,
)


@pytest.fixture
def redis_case(monkeypatch):
    url = os.environ.get('B29_TEST_REDIS_URL')
    if not url:
        pytest.skip('Set B29_TEST_REDIS_URL for real Redis integration tests')
    client = Redis.from_url(url, decode_responses=False,
                            socket_timeout=2, socket_connect_timeout=2)
    client.ping()  # An explicitly configured unavailable server fails the test.
    prefix = 'b29-test:' + secrets.token_hex(16) + ':v1:'
    monkeypatch.setattr(backend, '_PREFIX', prefix)
    store = backend.RedisLoginRateLimitStore(client=client)
    limiter = policy.LoginRateLimiter(store=store)
    originals = backend._ADMIT, backend._FINALIZE
    seconds, micros = client.time()
    origin = seconds + micros / 1_000_000

    def clock(offset):
        # Same server snapshot for every operation in exact boundary cases.
        # Redis's physical TTL clock continues advancing. Virtual-window cases
        # disable only minimum-remaining-TTL comparisons; corruption/TTL and all
        # concurrency cases exercise the unmodified production scripts.
        line = 'local now = tonumber(t[1]) + tonumber(t[2]) / 1000000'
        for attr, script in zip(('_ADMIT', '_FINALIZE'), originals):
            script = script.replace(line, f'local now = {origin + offset:.6f}')
            script = script.replace(
                'ttl < math.ceil((score + window - now) * 1000) - 2', 'false')
            script = script.replace(
                "redis.call('PTTL', KEYS[1]) < math.ceil((expiry - now) * 1000) - 2", 'false')
            monkeypatch.setattr(backend, attr, script)

    def keys(ip='192.0.2.1', email='person@example.com'):
        i, e, p = policy._derive_limiter_keys(client_ip=ip, email=email)
        return [prefix + 'ip:attempts:' + i.hex(),
                prefix + 'email:failures:' + e.hex(),
                prefix + 'pair:failures:' + p.hex(),
                prefix + 'email:open:' + e.hex(),
                prefix + 'pair:open:' + p.hex()]

    yield SimpleNamespace(client=client, store=store, limiter=limiter,
                          prefix=prefix, clock=clock, keys=keys)
    owned = list(client.scan_iter(match=prefix + '*'))
    if owned:
        client.delete(*owned)
    client.close()


def admit(case, ip='192.0.2.1', email='person@example.com'):
    return case.limiter.check(client_ip=ip, email=email)


def blocked(case, ip='192.0.2.1', email='person@example.com', wait=None):
    with pytest.raises(LoginRateLimitExceededError) as exc:
        admit(case, ip, email)
    assert type(exc.value.retry_after_seconds) is int
    if wait is not None:
        assert exc.value.retry_after_seconds == wait
    return exc.value.retry_after_seconds


@pytest.mark.parametrize('dimension,limit,window',
                         [('ip', 10, 300), ('email', 10, 900), ('pair', 5, 300)])
def test_exact_failure_threshold_and_sliding_window(redis_case, dimension, limit, window):
    c = redis_case
    c.clock(0)
    for n in range(limit):
        ip = f'192.0.2.{n+1}' if dimension == 'email' else '192.0.2.1'
        email = f'p{n}@example.com' if dimension == 'ip' else 'person@example.com'
        c.limiter.record_failure(permit=admit(c, ip, email))
    query_email = 'new@example.com' if dimension == 'ip' else 'person@example.com'
    blocked(c, email=query_email, wait=window)
    c.clock(window - 0.5)
    blocked(c, email=query_email, wait=1)
    c.clock(window)
    assert admit(c, email=query_email)


@pytest.mark.parametrize('dimension,limit', [('email', 10), ('pair', 5)])
def test_pending_reservations_and_blocked_admission_has_no_writes(redis_case, dimension, limit):
    c = redis_case
    c.clock(0)
    for n in range(limit):
        admit(c, f'192.0.2.{n+1}' if dimension == 'email' else '192.0.2.1')
    before = {key: c.client.dump(key) for key in c.client.scan_iter(match=c.prefix + '*')}
    blocked(c, wait=300)
    after = {key: c.client.dump(key) for key in c.client.scan_iter(match=c.prefix + '*')}
    assert before == after


def test_retry_after_uses_maximum_and_mixed_failure_pending_expirations(redis_case):
    c = redis_case
    c.clock(0)
    for n in range(9):
        c.limiter.record_failure(permit=admit(c, f'192.0.2.{n+1}'))
    c.clock(100)
    admit(c, '192.0.2.20')
    blocked(c, '192.0.2.21', wait=300)  # First capacity is the OPEN expiry.
    c.clock(400)
    p = admit(c, '192.0.2.21')
    c.limiter.record_failure(permit=p)
    blocked(c, '192.0.2.22', wait=500)  # Email's oldest failure expires at 900.


def test_retry_after_maximum_across_all_blocking_dimensions(redis_case):
    c = redis_case
    c.clock(0)
    for _ in range(5):
        c.limiter.record_failure(permit=admit(c))
    for n in range(2, 7):
        c.limiter.record_failure(permit=admit(c, f'192.0.2.{n}'))
    c.clock(100)
    for _ in range(5):
        admit(c, email=None)
    blocked(c, wait=800)  # IP/pair need 200 seconds; email needs 800.


@pytest.mark.parametrize('dimension,limit', [('email', 10), ('pair', 5)])
def test_protocol_supports_independently_optional_dimensions(redis_case, dimension, limit):
    c = redis_case
    i, e, pair = policy._derive_limiter_keys(client_ip='192.0.2.1', email='person@example.com')
    e, pair = (e, None) if dimension == 'email' else (None, pair)
    for _ in range(limit):
        token = secrets.token_urlsafe(32)
        assert c.store.admit_attempt(attempt_id=token, ip_key=i, email_key=e, pair_key=pair) is None
        c.store.record_failure(attempt_id=token)
    assert c.store.admit_attempt(attempt_id=secrets.token_urlsafe(32),
                                 ip_key=i, email_key=e, pair_key=pair) is not None
    assert not list(c.client.scan_iter(match=c.prefix +
                                      ('pair:*' if dimension == 'email' else 'email:*')))


def test_ip_history_expires_one_event_at_a_time(redis_case):
    c = redis_case
    c.clock(0)
    admit(c, email=None)
    c.clock(100)
    for _ in range(9):
        admit(c, email=None)
    c.clock(299.5)
    blocked(c, email=None, wait=1)
    c.clock(300)
    admit(c, email=None)
    blocked(c, email=None, wait=100)


@pytest.mark.parametrize('outcome', ['success', 'aborted'])
def test_neutral_terminal_releases_reservations_retains_failures_and_ip(redis_case, outcome):
    c = redis_case
    p = admit(c)
    c.limiter.record_failure(permit=p)
    ip, ef, pf, er, pr = c.keys()
    histories = c.client.dump(ef), c.client.dump(pf)
    p = admit(c)
    getattr(c.limiter, 'record_' + outcome)(permit=p)
    assert (c.client.dump(ef), c.client.dump(pf)) == histories
    assert c.client.zcard(er) == c.client.zcard(pr) == 0
    assert c.client.zcard(ip) == 2
    assert not c.client.exists(c.prefix + 'permit:' + p.attempt_id)


def test_failure_converts_reservations_to_full_window_history(redis_case):
    c = redis_case
    c.clock(0)
    p = admit(c)
    c.clock(299)
    c.limiter.record_failure(permit=p)
    _, ef, pf, er, pr = c.keys()
    assert c.client.zcard(er) == c.client.zcard(pr) == 0
    assert c.client.zcard(ef) == c.client.zcard(pf) == 1
    assert c.client.zscore(ef, p.attempt_id) == c.client.zscore(pf, p.attempt_id)
    assert 899000 <= c.client.pttl(ef) <= 900001
    assert 299000 <= c.client.pttl(pf) <= 300001
    c.clock(598.5)
    # New permits trigger pruning; the late failure still lives.
    p2 = admit(c)
    assert c.client.zcard(pf) == 1
    c.limiter.record_success(permit=p2)
    c.clock(599)
    admit(c)
    assert c.client.zcard(pf) == 0
    assert c.client.zcard(ef) == 1
    c.clock(1199)
    admit(c)
    assert c.client.zcard(ef) == 0


@pytest.mark.parametrize('first', ['failure', 'success', 'aborted'])
@pytest.mark.parametrize('second', ['failure', 'success', 'aborted'])
def test_duplicate_and_mixed_finalizations_fail_closed(redis_case, first, second):
    c = redis_case
    p = admit(c)
    getattr(c.limiter, 'record_' + first)(permit=p)
    with pytest.raises(LoginRateLimitInfrastructureError, match='^Login rate limiting unavailable$'):
        getattr(c.limiter, 'record_' + second)(permit=p)


@pytest.mark.parametrize('outcome', ['failure', 'success', 'aborted'])
def test_exact_expiry_rejected_and_reservations_reclaimed_independently(redis_case, outcome):
    c = redis_case
    c.clock(0)
    permits = [admit(c) for _ in range(5)]
    c.clock(300)
    with pytest.raises(LoginRateLimitInfrastructureError):
        getattr(c.limiter, 'record_' + outcome)(permit=permits[0])
    # Metadata exists, proving cleanup is independent of Redis permit TTL.
    assert c.client.exists(c.prefix + 'permit:' + permits[0].attempt_id)
    admit(c)
    _, _, _, er, pr = c.keys()
    assert c.client.zcard(er) == c.client.zcard(pr) == 1


def test_missing_expired_metadata_does_not_keep_reservations(redis_case):
    c = redis_case
    c.clock(0)
    p = admit(c)
    c.client.delete(c.prefix + 'permit:' + p.attempt_id)
    c.clock(300)
    admit(c)
    assert c.client.zcard(c.keys()[3]) == 1


@pytest.mark.parametrize('email', [None, '', '   '])
def test_source_only_identity_enforces_ip(redis_case, email):
    c = redis_case
    for _ in range(10):
        c.limiter.record_failure(permit=admit(c, email=email))
    blocked(c, email=email)
    assert not list(c.client.scan_iter(match=c.prefix + 'email:*'))
    assert not list(c.client.scan_iter(match=c.prefix + 'pair:*'))


def test_privacy_and_bounded_ttls(redis_case):
    c = redis_case
    permits = [admit(c) for _ in range(5)]
    c.limiter.record_failure(permit=permits[0])
    keys = list(c.client.scan_iter(match=c.prefix + '*'))
    assert len(keys) == 9  # Five dimensions and four OPEN permits.
    for key in keys:
        assert 0 < c.client.pttl(key) <= 900001
        data = key + c.client.dump(key)
        assert b'person@example.com' not in data
        assert b'192.0.2.1' not in data
    assert sum(c.client.zcard(k) for k in c.keys()) == 15


@pytest.mark.parametrize('dimension', range(5))
def test_wrong_key_type_fails_before_any_mutation(redis_case, dimension):
    c = redis_case
    p = admit(c)
    key = c.keys()[dimension]
    c.client.delete(key)
    c.client.set(key, b'corrupted')
    before = {k: c.client.dump(k) for k in c.client.scan_iter(match=c.prefix + '*')}
    with pytest.raises(LoginRateLimitInfrastructureError):
        admit(c)
    assert before == {k: c.client.dump(k) for k in before}
    if dimension:
        with pytest.raises(LoginRateLimitInfrastructureError):
            c.limiter.record_failure(permit=p)
        assert before == {k: c.client.dump(k) for k in before}


@pytest.mark.parametrize('corruption', ['metadata', 'score', 'member', 'ttl',
                                       'short_ttl', 'permit_ttl', 'reservation'])
def test_corrupted_state_finalization_has_no_partial_writes(redis_case, corruption):
    c = redis_case
    p = admit(c)
    permit = c.prefix + 'permit:' + p.attempt_id
    _, ef, pf, er, pr = c.keys()
    if corruption == 'metadata':
        c.client.hset(permit, 'state', 'FAILURE')
    elif corruption == 'score':
        c.client.zadd(er, {p.attempt_id: float('inf')})
    elif corruption == 'member':
        c.client.zadd(er, {'raw@example.com': 1})
    elif corruption == 'ttl':
        c.client.persist(er)
    elif corruption == 'short_ttl':
        c.client.pexpire(er, 100000)
    elif corruption == 'permit_ttl':
        c.client.pexpire(permit, 100000)
    else:
        c.client.zrem(pr, p.attempt_id)
    before = {k: c.client.dump(k) for k in c.client.scan_iter(match=c.prefix + '*')}
    with pytest.raises(LoginRateLimitInfrastructureError):
        c.limiter.record_failure(permit=p)
    assert before == {k: c.client.dump(k) for k in before}
    assert c.client.zcard(ef) == c.client.zcard(pf) == 0


@pytest.mark.parametrize('value', [None, '', 'bad', 'x' * 44, '../private', 1])
def test_invalid_permit_never_reaches_redis(value):
    client = Mock()
    store = backend.RedisLoginRateLimitStore(client=client)
    with pytest.raises(LoginRateLimitInfrastructureError):
        store.record_success(attempt_id=value)
    client.connection_pool.get_connection.assert_not_called()


def test_unknown_permit_fails_closed(redis_case):
    with pytest.raises(LoginRateLimitInfrastructureError):
        redis_case.store.record_aborted(attempt_id=secrets.token_urlsafe(32))


@pytest.mark.parametrize('response', [None, [], [1], [1, 1], [0, 0], [0, 901],
                                      [True, 0], [1, '0'], [2, 0], [1, 0, 0]])
def test_malformed_script_response_fails_closed(response):
    client = Mock()
    connection = client.connection_pool.get_connection.return_value
    connection.read_response.return_value = response
    store = backend.RedisLoginRateLimitStore(client=client)
    with pytest.raises(LoginRateLimitInfrastructureError) as exc:
        store.admit_attempt(attempt_id=secrets.token_urlsafe(32), ip_key=b'i' * 32,
                            email_key=None, pair_key=None)
    assert str(exc.value) == 'Login rate limiting unavailable'
    assert exc.value.__cause__ is None


@pytest.mark.parametrize('response', [None, [0, 1], [1, True], [1, 0, 0]])
def test_malformed_terminal_response_fails_closed(redis_case, monkeypatch, response):
    c = redis_case
    p = admit(c)
    command = c.store._command

    def malformed(*args):
        return response if args[0] == 'EVAL' else command(*args)

    monkeypatch.setattr(c.store, '_command', malformed)
    with pytest.raises(LoginRateLimitInfrastructureError):
        c.store.record_failure(attempt_id=p.attempt_id)


@pytest.mark.parametrize('error', [ConnectionError, TimeoutError, ResponseError])
def test_runtime_errors_sanitized_and_mutation_sent_only_once(error):
    client = Mock()
    connection = client.connection_pool.get_connection.return_value
    connection.read_response.side_effect = error('redis://secret@private:6379 identity')
    store = backend.RedisLoginRateLimitStore(client=client)
    with pytest.raises(LoginRateLimitInfrastructureError) as exc:
        store.admit_attempt(attempt_id=secrets.token_urlsafe(32), ip_key=b'i' * 32,
                            email_key=None, pair_key=None)
    assert str(exc.value) == 'Login rate limiting unavailable'
    assert exc.value.__cause__ is None
    assert connection.send_command.call_count == 1
    connection.retry.call_with_retry.assert_not_called()
    connection.disconnect.assert_called_once()
    client.connection_pool.release.assert_called_once_with(connection)


def test_real_unavailable_redis_fails_closed():
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        # Bound but not listening: no race with another process claiming port.
        client = Redis(host='127.0.0.1', port=listener.getsockname()[1],
                       socket_timeout=0.1, socket_connect_timeout=0.1)
        try:
            limiter = policy.LoginRateLimiter(store=backend.RedisLoginRateLimitStore(client=client))
            with pytest.raises(LoginRateLimitInfrastructureError):
                limiter.check(client_ip='192.0.2.1', email='person@example.com')
        finally:
            client.close()


@pytest.mark.parametrize('operation', ['admission', 'failure', 'success', 'aborted'])
def test_ambiguous_timeout_after_real_mutation_is_not_replayed(redis_case, monkeypatch, operation):
    c = redis_case
    p = None if operation == 'admission' else admit(c)
    connection = c.client.connection_pool.get_connection()
    c.client.connection_pool.release(connection)
    send, read = connection.send_command, connection.read_response
    commands = []

    def send_once(*args, **kwargs):
        commands.append(args[0])
        return send(*args, **kwargs)

    def lose_reply(*args, **kwargs):
        result = read(*args, **kwargs)
        if commands[-1] == 'EVAL':
            # The real server already applied the mutation; its reply is lost.
            raise TimeoutError('private ambiguous reply')
        return result

    with monkeypatch.context() as patch:
        patch.setattr(connection, 'send_command', send_once)
        patch.setattr(connection, 'read_response', lose_reply)
        with pytest.raises(LoginRateLimitInfrastructureError):
            if operation == 'admission':
                admit(c)
            else:
                getattr(c.limiter, 'record_' + operation)(permit=p)
    assert commands.count('EVAL') == 1
    ip, ef, pf, er, pr = c.keys()
    assert c.client.zcard(ip) == 1
    assert c.client.zcard(er) == c.client.zcard(pr) == int(operation == 'admission')
    assert c.client.zcard(ef) == c.client.zcard(pf) == int(operation == 'failure')


def test_normalized_email_and_independent_limiters_share_history(redis_case):
    c = redis_case
    other = policy.LoginRateLimiter(store=backend.RedisLoginRateLimitStore(client=c.client))
    for n in range(10):
        p = c.limiter.check(client_ip=f'192.0.2.{n+1}', email=' PERSON@Example.COM ')
        other.record_failure(permit=p)
    with pytest.raises(LoginRateLimitExceededError):
        other.check(client_ip='192.0.2.20', email='person@example.com')


def test_attempt_collision_fails_before_writing(redis_case):
    c = redis_case
    p = admit(c)
    c.limiter.record_success(permit=p)
    i, e, pair = policy._derive_limiter_keys(client_ip='192.0.2.1', email='person@example.com')
    with pytest.raises(LoginRateLimitInfrastructureError):
        c.store.admit_attempt(attempt_id=p.attempt_id, ip_key=i, email_key=e, pair_key=pair)
    assert c.client.zcard(c.keys()[0]) == 1
    assert not c.client.exists(c.prefix + 'permit:' + p.attempt_id)


def test_idle_state_is_reclaimed_by_redis_ttls(redis_case):
    import time
    c = redis_case
    c.limiter.record_failure(permit=admit(c))
    admit(c)
    owned = list(c.client.scan_iter(match=c.prefix + '*'))
    assert all(c.client.pttl(key) > 0 for key in owned)
    # Compress only test key TTLs to verify Redis automatic idle reclamation.
    for key in owned:
        c.client.pexpire(key, 5)
    deadline = time.monotonic() + 2
    while c.client.exists(*owned) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not list(c.client.scan_iter(match=c.prefix + '*'))
