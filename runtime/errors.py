"""Stable errors shared by the runtime and API boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A safe terminal error returned to callers."""

    code: str
    message: str


class TinyAgentError(Exception):
    """Base class for expected failures."""

    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.safe_message = message

    def as_info(self) -> ErrorInfo:
        """Convert the exception into its public representation."""
        return ErrorInfo(code=self.code, message=self.safe_message)


class SessionNotFoundError(TinyAgentError):
    """Raised when a session identifier is unknown."""

    code = "session_not_found"


class SessionOwnershipError(TinyAgentError):
    """Raised when a user attempts to access another user's session."""

    code = "session_forbidden"


class SessionExistsError(TinyAgentError):
    """Raised when a requested session identifier already exists."""

    code = "session_exists"


class InvalidInputError(TinyAgentError):
    """Raised for invalid public input."""

    code = "invalid_input"

