"""Per-browser conversation state for the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from frontend.client import TraceEntry


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One locally rendered message and the trace associated with it."""

    role: Literal["user", "assistant"]
    content: str
    trace: tuple[TraceEntry, ...] = ()
    is_error: bool = False


@dataclass(slots=True)
class Conversation:
    """One frontend conversation mapped to an optional runtime session."""

    id: str
    title: str = "New conversation"
    title_revision: int = 0
    is_untitled: bool = True
    api_session_id: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)

    def title_from_prompt(self, prompt: str) -> None:
        """Use the first prompt as a compact title while preserving manual titles."""
        if not self.is_untitled:
            return
        words = " ".join(prompt.split())
        self.title = f"{words[:45].rstrip()}…" if len(words) > 45 else words
        self.title_revision += 1
        self.is_untitled = False


@dataclass(slots=True)
class ConversationState:
    """Manage conversations and keep exactly one of them active."""

    conversations: list[Conversation] = field(default_factory=list)
    active_id: str = ""

    @classmethod
    def create(cls) -> ConversationState:
        """Create state with one active draft conversation."""
        state = cls()
        state.new_conversation()
        return state

    @property
    def active(self) -> Conversation:
        """Return the active conversation, repairing an empty collection if needed."""
        for conversation in self.conversations:
            if conversation.id == self.active_id:
                return conversation
        return self.new_conversation()

    def new_conversation(self) -> Conversation:
        """Create and select a new draft conversation."""
        conversation = Conversation(id=uuid4().hex)
        self.conversations.insert(0, conversation)
        self.active_id = conversation.id
        return conversation

    def select(self, conversation_id: str) -> None:
        """Select an existing conversation by its frontend identifier."""
        if not any(item.id == conversation_id for item in self.conversations):
            raise ValueError("Conversation does not exist.")
        self.active_id = conversation_id

    def rename(self, conversation_id: str, title: str) -> None:
        """Rename a conversation after normalizing and validating its title."""
        normalized = " ".join(title.split())
        if not normalized:
            raise ValueError("Conversation name cannot be blank.")
        for conversation in self.conversations:
            if conversation.id == conversation_id:
                conversation.title = normalized[:80]
                conversation.title_revision += 1
                conversation.is_untitled = False
                return
        raise ValueError("Conversation does not exist.")

    def remove(self, conversation_id: str) -> None:
        """Forget one local conversation and select a neighboring record."""
        removed_index = next(
            (
                index
                for index, conversation in enumerate(self.conversations)
                if conversation.id == conversation_id
            ),
            None,
        )
        if removed_index is None:
            raise ValueError("Conversation does not exist.")
        removed = self.conversations.pop(removed_index)
        if removed.id != self.active_id:
            return
        if not self.conversations:
            self.new_conversation()
            return
        next_index = min(removed_index, len(self.conversations) - 1)
        self.active_id = self.conversations[next_index].id

    def reset(self) -> None:
        """Discard all frontend conversations and create a fresh active record."""
        self.conversations.clear()
        self.active_id = ""
        self.new_conversation()
