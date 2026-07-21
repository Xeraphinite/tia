"""Small synchronous client for Tiny Agent's HTTP boundary."""

from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

from runtime.contracts import JSONObject


class TraceEntry(BaseModel):
    """One sanitized execution event returned by the API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    kind: str
    data: JSONObject


class APIError(BaseModel):
    """A stable, user-safe API error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str


class MessageResult(BaseModel):
    """The terminal result for one user turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    status: str
    answer: str | None
    error: APIError | None
    trace: tuple[TraceEntry, ...]


class _CreateSessionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str


class _ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    detail: APIError


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """Raw HTTP response at the narrow transport boundary."""

    status: int
    body: bytes


class HTTPTransport(Protocol):
    """Minimal replaceable transport used by the frontend client."""

    def post(self, url: str, payload: bytes, timeout_seconds: float) -> TransportResponse:
        """Send JSON and return the status and raw body."""
        ...


class UrllibTransport:
    """Standard-library JSON transport with bounded requests."""

    def post(self, url: str, payload: bytes, timeout_seconds: float) -> TransportResponse:
        request = Request(
            url,
            data=payload,
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            response = cast(HTTPResponse, urlopen(request, timeout=timeout_seconds))
            with response:
                return TransportResponse(status=response.status, body=response.read())
        except HTTPError as exc:
            with exc:
                return TransportResponse(status=exc.code, body=exc.read())
        except URLError as exc:
            raise FrontendAPIError(
                "api_unreachable",
                "The Tiny Agent API is not reachable. Start the API and try again.",
            ) from exc
        except TimeoutError as exc:
            raise FrontendAPIError(
                "api_timeout", "The Tiny Agent API did not respond before the timeout."
            ) from exc


class FrontendAPIError(Exception):
    """Safe failure suitable for display in the frontend."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class TinyAgentClient:
    """Create sessions and submit turns to a running Tiny Agent API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 70.0,
        transport: HTTPTransport | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API URL must be an absolute HTTP or HTTPS URL.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = normalized
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibTransport()

    def create_session(self, user_id: str) -> str:
        """Create a new API-owned conversation window."""
        response = self._post("/v1/sessions", {"user_id": user_id})
        try:
            return _CreateSessionResult.model_validate_json(response.body).session_id
        except ValidationError as exc:
            raise self._invalid_response() from exc

    def send_message(self, user_id: str, session_id: str, message: str) -> MessageResult:
        """Submit one turn and parse its terminal result."""
        response = self._post(
            f"/v1/sessions/{session_id}/messages",
            {"user_id": user_id, "message": message},
        )
        try:
            return MessageResult.model_validate_json(response.body)
        except ValidationError as exc:
            raise self._invalid_response() from exc

    def _post(self, path: str, payload: JSONObject) -> TransportResponse:
        import json

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response = self._transport.post(
            f"{self._base_url}{path}", body, self._timeout_seconds
        )
        if 200 <= response.status < 300:
            return response
        try:
            error = _ErrorEnvelope.model_validate_json(response.body).detail
        except ValidationError as exc:
            raise FrontendAPIError(
                "api_error", f"The Tiny Agent API returned HTTP {response.status}."
            ) from exc
        raise FrontendAPIError(error.code, error.message)

    @staticmethod
    def _invalid_response() -> FrontendAPIError:
        return FrontendAPIError(
            "invalid_api_response", "The Tiny Agent API returned an invalid response."
        )
