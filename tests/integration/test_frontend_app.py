"""Integration tests for Streamlit session controls."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from streamlit.testing.v1 import AppTest

from frontend.sessions import ConversationState

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"
pytestmark = pytest.mark.integration


def test_sidebar_creates_renames_switches_and_removes_conversations() -> None:
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    state = cast(ConversationState, app.session_state["conversation_state"])
    first_id = state.active_id

    new_button = next(button for button in app.button if button.label == "New conversation")
    app = new_button.click().run()
    state = cast(ConversationState, app.session_state["conversation_state"])
    second_id = state.active_id
    assert second_id != first_id
    assert len(state.conversations) == 2

    title_input = app.text_input(f"conversation-title-{second_id}-0")
    title_input.set_value("Project roadmap")
    rename_button = next(button for button in app.button if button.label == "Rename")
    app = rename_button.click().run()
    state = cast(ConversationState, app.session_state["conversation_state"])
    assert state.active.title == "Project roadmap"

    app = app.button(f"select-conversation-{first_id}").click().run()
    state = cast(ConversationState, app.session_state["conversation_state"])
    assert state.active_id == first_id

    app = app.button(f"remove-conversation-{first_id}").click().run()
    state = cast(ConversationState, app.session_state["conversation_state"])
    assert state.active_id == second_id
    assert [conversation.title for conversation in state.conversations] == [
        "Project roadmap"
    ]
    assert not app.exception
