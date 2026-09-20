from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


BASE_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
STRICT_TRANSPORT_SECURITY = "max-age=31536000; includeSubDomains"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, include_hsts: bool) -> None:
        super().__init__(app)
        self.include_hsts = include_hsts

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for header_name, header_value in BASE_SECURITY_HEADERS.items():
            response.headers[header_name] = header_value
        if self.include_hsts:
            response.headers["Strict-Transport-Security"] = (
                STRICT_TRANSPORT_SECURITY
            )
        return response
