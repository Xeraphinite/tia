"""Unit tests for frontend conversation session management."""

from __future__ import annotations

import pytest

from frontend.sessions import ChatMessage, ConversationState

pytestmark = pytest.mark.unit


def test_conversations_preserve_isolated_messages_and_runtime_sessions() -> None:
    state = ConversationState.create()
    first = state.active
    first.api_session_id = "runtime-one"
    first.messages.append(ChatMessage(role="user", content="First"))

    second = state.new_conversation()
    second.api_session_id = "runtime-two"
    second.messages.append(ChatMessage(role="user", content="Second"))
    state.select(first.id)

    assert state.active.api_session_id == "runtime-one"
    assert [message.content for message in state.active.messages] == ["First"]
    state.select(second.id)
    assert [message.content for message in state.active.messages] == ["Second"]


def test_conversation_titles_support_automatic_and_manual_names() -> None:
    state = ConversationState.create()

    state.active.title_from_prompt("  Explain   Tiny Agent session management  ")
    assert state.active.title == "Explain Tiny Agent session management"

    state.rename(state.active_id, "  Project   notes  ")
    state.active.title_from_prompt("This must not replace the manual title")
    assert state.active.title == "Project notes"

    state.rename(state.active_id, "New conversation")
    state.active.title_from_prompt("A manual default-looking title is still manual")
    assert state.active.title == "New conversation"

    with pytest.raises(ValueError, match="cannot be blank"):
        state.rename(state.active_id, "   ")


def test_removing_active_conversation_selects_another_or_creates_a_draft() -> None:
    state = ConversationState.create()
    first_id = state.active_id
    second_id = state.new_conversation().id

    state.remove(second_id)
    assert state.active_id == first_id
    assert len(state.conversations) == 1

    state.remove(first_id)
    assert len(state.conversations) == 1
    assert state.active.title == "New conversation"
    assert state.active_id != first_id
