"""Deterministic test doubles for orchestration tests."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence

from runtime.contracts import Message, ModelClientError, ModelResponse, ToolDefinition


class ScriptedModel:
    """Return a fixed sequence while recording complete model inputs."""

    def __init__(self, responses: Sequence[ModelResponse | ModelClientError]) -> None:
        self._responses = deque(responses)
        self.calls: list[tuple[tuple[Message, ...], tuple[ToolDefinition, ...]]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        self.calls.append((tuple(messages), tuple(tools)))
        if not self._responses:
            raise AssertionError("scripted model received an unexpected call")
        response = self._responses.popleft()
        if isinstance(response, ModelClientError):
            raise response
        return response


class SleepingModel:
    """Track concurrent model calls and return a final response."""

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        del messages, tools
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            return ModelResponse(content="done")
        finally:
            self.active -= 1
