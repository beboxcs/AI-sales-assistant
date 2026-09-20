from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.auth.human.login_rate_limit import LoginRateLimiter
from app.auth.human.redis_login_rate_limit_store import RedisLoginRateLimitStore
from app.auth.human.exceptions import (
    LoginRateLimitExceededError, LoginRateLimitInfrastructureError,
)
from test_redis_login_rate_limit_store import redis_case


@pytest.mark.parametrize('dimension,limit', [('pair', 5), ('email', 10), ('ip', 10)])
def test_independent_workers_cannot_bypass_limits(redis_case, dimension, limit):
    c = redis_case
    barrier = Barrier(24)

    def worker(n):
        # Independent stores and clients/pools share only the Redis server.
        from redis import Redis
        client = Redis(connection_pool=c.client.connection_pool.__class__(
            **c.client.connection_pool.connection_kwargs))
        try:
            limiter = LoginRateLimiter(store=RedisLoginRateLimitStore(client=client))
            barrier.wait(timeout=10)
            ip = f'192.0.2.{n+1}' if dimension == 'email' else '192.0.2.1'
            email = f'p{n}@example.com' if dimension == 'ip' else 'person@example.com'
            try:
                return limiter.check(client_ip=ip, email=email)
            except LoginRateLimitExceededError:
                return None
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(worker, range(24)))
    admitted = [p for p in results if p is not None]
    assert len(admitted) == limit
    assert len(set(p.attempt_id for p in admitted)) == limit
    assert len(list(c.client.scan_iter(match=c.prefix + 'permit:*'))) == limit
    assert sum(c.client.zcard(key) for key in c.client.scan_iter(
        match=c.prefix + 'ip:attempts:*')) == limit


def test_concurrent_mixed_finalization_exactly_one_succeeds(redis_case):
    c = redis_case
    permit = c.limiter.check(client_ip='192.0.2.1', email='person@example.com')
    barrier = Barrier(24)

    def worker(n):
        limiter = LoginRateLimiter(store=RedisLoginRateLimitStore(client=c.client))
        outcome = ('failure', 'success', 'aborted')[n % 3]
        barrier.wait(timeout=10)
        try:
            getattr(limiter, 'record_' + outcome)(permit=permit)
            return outcome
        except LoginRateLimitInfrastructureError:
            return None

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(worker, range(24)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    _, ef, pf, er, pr = c.keys()
    assert c.client.zcard(er) == c.client.zcard(pr) == 0
    assert c.client.zcard(ef) == c.client.zcard(pf) == int(winners == ['failure'])
    assert not c.client.exists(c.prefix + 'permit:' + permit.attempt_id)
