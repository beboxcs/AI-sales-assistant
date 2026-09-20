"""Single-Redis login accounting using injected clients and atomic Lua scripts.

No cluster support: all script keys belong to the frozen single-server topology.
Presence of validated permit metadata represents OPEN; deletion consumes it.
"""

import re

from redis import Redis

from app.auth.human.exceptions import LoginRateLimitInfrastructureError

_MESSAGE = "Login rate limiting unavailable"
_PREFIX = "login-rate-limit:v1:"
_ATTEMPT = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_IDENTITY = re.compile(r"[0-9a-f]{64}\Z")

# Scores are Redis TIME seconds, including microseconds. Validate every touched
# structure before pruning: Lua errors do not roll back earlier writes.
_COMMON = r"""
local function invalid() error('invalid limiter state', 0) end
local function kind(key) return redis.call('TYPE', key).ok end
local function token(s)
    return #s == 43 and string.match(s, '^[%w_-]+$') ~= nil
end
local function identity(s)
    return s == '' or (#s == 64 and string.match(s, '^[0-9a-f]+$') ~= nil)
end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local function events(key, maximum, future, window)
    local k = kind(key)
    if k == 'none' then return {} end
    local ttl = redis.call('PTTL', key)
    if k ~= 'zset' or ttl < 0 then invalid() end
    if redis.call('ZCARD', key) > maximum then invalid() end
    local rows = redis.call('ZRANGE', key, 0, -1, 'WITHSCORES')
    for i = 1, #rows, 2 do
        local score = tonumber(rows[i+1])
        if not token(rows[i]) or not score or score < 0 or
            score ~= score or score > now + future or
            ttl < math.ceil((score + window - now) * 1000) - 2 then invalid() end
    end
    return rows
end
local function disjoint(a, b)
    local members = {}
    for i = 1, #a, 2 do members[a[i]] = true end
    for i = 1, #b, 2 do if members[b[i]] then invalid() end end
end
local function prune(key, cutoff)
    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
end
local function duration(history, window, pending, limit)
    local expirations = {}
    for i = 2, #history, 2 do
        local expiry = tonumber(history[i]) + window
        if expiry > now then table.insert(expirations, expiry) end
    end
    for i = 2, #pending, 2 do
        local expiry = tonumber(pending[i])
        if expiry > now then table.insert(expirations, expiry) end
    end
    if #expirations > limit then invalid() end
    if #expirations < limit then return 0 end
    table.sort(expirations)
    return expirations[#expirations - limit + 1] - now
end
local function add(key, score, member, ttl)
    redis.call('ZADD', key, score, member)
    redis.call('PEXPIRE', key, ttl)
end
"""

_ADMIT = _COMMON + r"""
if #KEYS ~= 6 or #ARGV ~= 3 or not token(ARGV[1]) or
    not identity(ARGV[2]) or not identity(ARGV[3]) then invalid() end
if kind(KEYS[6]) ~= 'none' then invalid() end
local ip = events(KEYS[1], 10, 0, 300)
local ef, pf, er, pr = {}, {}, {}, {}
if ARGV[2] ~= '' then
    ef = events(KEYS[2], 10, 0, 900)
    er = events(KEYS[4], 10, 300, 0)
    disjoint(ef, er)
    if redis.call('ZSCORE', KEYS[2], ARGV[1]) or
        redis.call('ZSCORE', KEYS[4], ARGV[1]) then invalid() end
end
if ARGV[3] ~= '' then
    pf = events(KEYS[3], 5, 0, 300)
    pr = events(KEYS[5], 5, 300, 0)
    disjoint(pf, pr)
    if redis.call('ZSCORE', KEYS[3], ARGV[1]) or
        redis.call('ZSCORE', KEYS[5], ARGV[1]) then invalid() end
end
if redis.call('ZSCORE', KEYS[1], ARGV[1]) then invalid() end
local wait = duration(ip, 300, {}, 10)
if ARGV[2] ~= '' then
    wait = math.max(wait, duration(ef, 900, er, 10))
end
if ARGV[3] ~= '' then
    wait = math.max(wait, duration(pf, 300, pr, 5))
end
-- All validation is complete; only now perform cleanup/mutations.
prune(KEYS[1], now - 300)
if ARGV[2] ~= '' then
    prune(KEYS[2], now - 900)
    prune(KEYS[4], now)
end
if ARGV[3] ~= '' then
    prune(KEYS[3], now - 300)
    prune(KEYS[5], now)
end
if wait > 0 then return {0, math.max(1, math.ceil(wait))} end
local expiry = now + 300
add(KEYS[1], now, ARGV[1], 300001)
redis.call('HSET', KEYS[6], 'version', '1', 'state', 'OPEN',
    'email', ARGV[2], 'pair', ARGV[3], 'expiry', string.format('%.6f', expiry))
redis.call('PEXPIRE', KEYS[6], 300001)
if ARGV[2] ~= '' then
    add(KEYS[4], expiry, ARGV[1], 300001)
end
if ARGV[3] ~= '' then
    add(KEYS[5], expiry, ARGV[1], 300001)
end
return {1, 0}
"""

_FINALIZE = _COMMON + r"""
if #KEYS ~= 5 or #ARGV ~= 5 or not token(ARGV[1]) or
    not identity(ARGV[2]) or not identity(ARGV[3]) or
    (ARGV[5] ~= 'failure' and ARGV[5] ~= 'success' and ARGV[5] ~= 'aborted')
    then invalid() end
if kind(KEYS[1]) ~= 'hash' or redis.call('HLEN', KEYS[1]) ~= 5 or
    redis.call('PTTL', KEYS[1]) < 0 then invalid() end
local p = redis.call('HMGET', KEYS[1], 'version', 'state', 'email', 'pair', 'expiry')
local expiry = tonumber(p[5])
if p[1] ~= '1' or p[2] ~= 'OPEN' or p[3] ~= ARGV[2] or
    p[4] ~= ARGV[3] or p[5] ~= ARGV[4] or not expiry or
    expiry ~= expiry or expiry <= now or expiry > now + 300 or
    redis.call('PTTL', KEYS[1]) < math.ceil((expiry - now) * 1000) - 2
    then invalid() end
if ARGV[2] ~= '' then
    local ef = events(KEYS[2], 10, 0, 900)
    local er = events(KEYS[4], 10, 300, 0)
    disjoint(ef, er)
    duration(ef, 900, er, 10)
    local es = tonumber(redis.call('ZSCORE', KEYS[4], ARGV[1]))
    -- Hash serialization and Redis double scores may differ below a microsecond.
    if not es or math.abs(es - expiry) > 0.000001 then invalid() end
    if redis.call('ZSCORE', KEYS[2], ARGV[1]) then invalid() end
end
if ARGV[3] ~= '' then
    local pf = events(KEYS[3], 5, 0, 300)
    local pr = events(KEYS[5], 5, 300, 0)
    disjoint(pf, pr)
    duration(pf, 300, pr, 5)
    local ps = tonumber(redis.call('ZSCORE', KEYS[5], ARGV[1]))
    if not ps or math.abs(ps - expiry) > 0.000001 then invalid() end
    if redis.call('ZSCORE', KEYS[3], ARGV[1]) then invalid() end
end
-- Consume exactly once and atomically replace reservations on failure.
redis.call('DEL', KEYS[1])
if ARGV[2] ~= '' then
    prune(KEYS[2], now - 900)
    prune(KEYS[4], now)
    redis.call('ZREM', KEYS[4], ARGV[1])
    if ARGV[5] == 'failure' then
        add(KEYS[2], now, ARGV[1], 900001)
    end
end
if ARGV[3] ~= '' then
    prune(KEYS[3], now - 300)
    prune(KEYS[5], now)
    redis.call('ZREM', KEYS[5], ARGV[1])
    if ARGV[5] == 'failure' then
        add(KEYS[3], now, ARGV[1], 300001)
    end
end
return {1, 0}
"""


class RedisLoginRateLimitStore:
    """Implements LoginRateLimitStore without configuration or client creation."""

    def __init__(self, *, client: Redis):
        self._client = client

    def _command(self, *args):
        """Send once: redis-py 8.1 execute_command retries ambiguous mutations.

        Acquire an exclusive pooled connection, bypass the command retry wrapper,
        and discard its socket on any error. Pool connection/handshake retries
        precede the mutation and cannot replay it. Do not change shared client
        retry settings. This path targets the frozen synchronous redis-py client.
        """
        pool = self._client.connection_pool
        connection = None
        try:
            connection = pool.get_connection()
            connection.send_command(*args, check_health=False)
            return connection.read_response()
        except Exception:
            if connection is not None:
                connection.disconnect()
            raise LoginRateLimitInfrastructureError(_MESSAGE) from None
        finally:
            if connection is not None:
                pool.release(connection)

    @staticmethod
    def _attempt(attempt_id):
        if not isinstance(attempt_id, str) or not _ATTEMPT.fullmatch(attempt_id):
            raise ValueError
        return _PREFIX + 'permit:' + attempt_id

    @staticmethod
    def _identity(value):
        if value is None:
            return ''
        if not isinstance(value, bytes) or len(value) != 32:
            raise ValueError
        return value.hex()

    @staticmethod
    def _dimensions(email, pair, permit):
        # Absent dimensions are never accessed in Lua; pass explicit KEYS.
        return [
            _PREFIX + 'email:failures:' + email if email else permit,
            _PREFIX + 'pair:failures:' + pair if pair else permit,
            _PREFIX + 'email:open:' + email if email else permit,
            _PREFIX + 'pair:open:' + pair if pair else permit,
        ]

    def _eval(self, script, keys, args):
        response = self._command('EVAL', script, len(keys), *keys, *args)
        if (not isinstance(response, list) or len(response) != 2 or
                any(type(item) is not int for item in response)):
            raise ValueError
        return response

    def admit_attempt(self, *, attempt_id: str, ip_key: bytes,
                      email_key: bytes | None, pair_key: bytes | None) -> int | None:
        try:
            permit = self._attempt(attempt_id)
            ip = self._identity(ip_key)
            if not ip:
                raise ValueError
            email, pair = self._identity(email_key), self._identity(pair_key)
            keys = [_PREFIX + 'ip:attempts:' + ip,
                    *self._dimensions(email, pair, permit), permit]
            status, wait = self._eval(_ADMIT, keys, [attempt_id, email, pair])
            if status == 1 and wait == 0:
                return None
            if status == 0 and 1 <= wait <= 900:
                return wait
            raise ValueError
        except Exception:
            raise LoginRateLimitInfrastructureError(_MESSAGE) from None

    def _finalize(self, attempt_id, outcome):
        try:
            permit = self._attempt(attempt_id)
            # Read only to discover explicitly declared KEYS. The atomic script
            # revalidates the entire snapshot and OPEN state before any mutation.
            raw = self._command('HGETALL', permit)
            if isinstance(raw, dict):  # RESP3
                metadata = raw
            elif isinstance(raw, list) and len(raw) == 10:
                metadata = dict(zip(raw[::2], raw[1::2]))
            else:
                raise ValueError
            if len(metadata) != 5:
                raise ValueError
            metadata = {k.decode('ascii'): v.decode('ascii')
                        for k, v in metadata.items()}
            if set(metadata) != {'version', 'state', 'email', 'pair', 'expiry'}:
                raise ValueError
            email, pair = metadata['email'], metadata['pair']
            if any(value and not _IDENTITY.fullmatch(value) for value in (email, pair)):
                raise ValueError
            keys = [permit, *self._dimensions(email, pair, permit)]
            response = self._eval(_FINALIZE, keys,
                                  [attempt_id, email, pair, metadata['expiry'], outcome])
            if response != [1, 0]:
                raise ValueError
        except Exception:
            raise LoginRateLimitInfrastructureError(_MESSAGE) from None

    def record_failure(self, *, attempt_id: str) -> None:
        self._finalize(attempt_id, 'failure')

    def record_success(self, *, attempt_id: str) -> None:
        self._finalize(attempt_id, 'success')

    def record_aborted(self, *, attempt_id: str) -> None:
        self._finalize(attempt_id, 'aborted')
