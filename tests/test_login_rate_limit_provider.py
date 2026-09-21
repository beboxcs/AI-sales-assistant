import inspect
from unittest.mock import Mock

import pytest

from app.auth.human import login_rate_limit as policy
from app.auth.human import redis_login_rate_limit_store as backend
from app.auth.human.exceptions import LoginRateLimitInfrastructureError
from app.auth.human.types import LoginRateLimitPermit
from app.infrastructure import redis as infrastructure
from test_redis_login_rate_limit_store import redis_case


@pytest.mark.parametrize('env', ['local', 'test', 'staging', 'production'])
def test_provider_injects_frozen_factory_client_without_connecting(monkeypatch, env):
    monkeypatch.setattr(policy.settings, 'APP_ENV', env)
    monkeypatch.setattr(policy.settings, 'LOGIN_RATE_LIMIT_BACKEND', 'redis')
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(infrastructure, 'create_redis_client', factory)
    provider = policy._DefaultLoginRateLimiterProvider()
    limiter = provider.get()
    assert limiter is provider.get()
    assert isinstance(limiter._store, backend.RedisLoginRateLimitStore)
    assert limiter._store._client is client
    factory.assert_called_once_with(policy.settings)
    client.assert_not_called()
    client.connection_pool.get_connection.assert_not_called()
    client.ping.assert_not_called()


@pytest.mark.parametrize('env', ['local', 'test'])
@pytest.mark.parametrize('choice', [None, 'memory'])
def test_local_defaults_preserve_memory(monkeypatch, env, choice):
    monkeypatch.setattr(policy.settings, 'APP_ENV', env)
    monkeypatch.setattr(policy.settings, 'LOGIN_RATE_LIMIT_BACKEND', choice)
    provider = policy._DefaultLoginRateLimiterProvider()
    assert isinstance(provider.get()._store, policy.InMemoryLoginRateLimitStore)
    assert provider.get() is provider.get()


def test_runtime_redis_outage_has_no_memory_fallback(monkeypatch):
    monkeypatch.setattr(policy.settings, 'LOGIN_RATE_LIMIT_BACKEND', 'redis')
    client = Mock()
    client.connection_pool.get_connection.side_effect = RuntimeError('redis://private')
    monkeypatch.setattr(infrastructure, 'create_redis_client', Mock(return_value=client))
    provider = policy._DefaultLoginRateLimiterProvider()
    monkeypatch.setattr(policy, '_default_provider', provider)
    for _ in range(2):
        with pytest.raises(LoginRateLimitInfrastructureError) as exc:
            policy.check_login_rate_limit(client_ip='192.0.2.1', email='person@example.com')
        assert str(exc.value) == 'Login rate limiting unavailable'
        assert exc.value.__cause__ is None
    assert provider._local_limiter is None
    assert isinstance(provider.get()._store, backend.RedisLoginRateLimitStore)


def test_factory_errors_sanitized(monkeypatch):
    monkeypatch.setattr(policy.settings, 'LOGIN_RATE_LIMIT_BACKEND', 'redis')
    monkeypatch.setattr(infrastructure, 'create_redis_client',
                        Mock(side_effect=RuntimeError('redis://private')))
    with pytest.raises(LoginRateLimitInfrastructureError, match='^Login rate limiting unavailable$'):
        policy._DefaultLoginRateLimiterProvider().get()


def test_public_signatures_and_permit_contract_unchanged():
    assert list(inspect.signature(policy.check_login_rate_limit).parameters) == ['client_ip', 'email']
    for method in (policy.record_login_failure, policy.record_login_success, policy.record_login_aborted):
        assert list(inspect.signature(method).parameters) == ['permit']
        assert inspect.signature(method).parameters['permit'].kind == inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(LoginRateLimitPermit).parameters) == ['attempt_id']
    assert list(inspect.signature(policy.LoginRateLimiter).parameters) == ['store']


@pytest.mark.parametrize('scenario', ['threshold', 'outage'])
def test_login_api_uses_redis_via_unchanged_adapter(redis_case, monkeypatch, scenario):
    from fastapi.testclient import TestClient
    from app.api.v1.auth import get_human_auth_service
    from app.auth.human.exceptions import InvalidCredentialsError
    from app.main import app
    from test_human_auth_login_api import _human_auth_service, _login_payload

    c = redis_case
    monkeypatch.setattr(policy.settings, 'LOGIN_RATE_LIMIT_BACKEND', 'redis')
    monkeypatch.setattr(policy.settings, 'TRUSTED_BROWSER_ORIGINS', ('https://app.example.com',))
    monkeypatch.setattr(infrastructure, 'create_redis_client', Mock(return_value=c.client))
    provider = policy._DefaultLoginRateLimiterProvider()
    monkeypatch.setattr(policy, '_default_provider', provider)
    service = _human_auth_service(auth_error=InvalidCredentialsError('private credentials'))
    original = app.dependency_overrides.copy()
    app.dependency_overrides[get_human_auth_service] = lambda: service
    try:
        with TestClient(app, client=('192.0.2.1', 50000)) as client:
            def login():
                return client.post('/api/v1/auth/login',
                                   json=_login_payload(
                                       email='person@example.com',
                                       password='private-password',
                                   ),
                                   headers={'Origin': 'https://app.example.com'})
            if scenario == 'outage':
                with monkeypatch.context() as patch:
                    patch.setattr(c.client.connection_pool, 'get_connection',
                                  Mock(side_effect=RuntimeError('redis://private')))
                    response = login()
                assert response.status_code == 500
                assert response.json()['error']['code'] == 'AUTHENTICATION_UNAVAILABLE'
                assert service.credential_calls == []
                assert provider._local_limiter is None
            else:
                c.clock(0)
                for _ in range(5):
                    assert login().status_code == 401
                response = login()
                assert response.status_code == 429
                assert response.headers['Retry-After'] == '300'
                assert len(service.credential_calls) == 5
            assert 'redis://' not in response.text
            assert 'person@example.com' not in response.text
            assert 'private-password' not in response.text
    finally:
        app.dependency_overrides = original
