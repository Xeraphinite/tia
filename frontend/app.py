"""Streamlit chat interface for a running Tiny Agent API."""

from __future__ import annotations

import os
from typing import cast

import streamlit as st

from frontend.client import FrontendAPIError, TinyAgentClient, TraceEntry
from frontend.sessions import ChatMessage, ConversationState

SUGGESTIONS = (
    ("Weather", "What's the weather in Shanghai?"),
    ("Calculate", "Calculate (42 * 17) + 9."),
    ("Plan", "Add 'Review agent traces' to my todo list."),
)


def _initialize_state() -> None:
    defaults: dict[str, object] = {
        "active_api_url": os.getenv("TIA_API_URL", "http://127.0.0.1:8000"),
        "active_user_id": os.getenv("TIA_USER_ID", "local-user"),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "conversation_state" not in st.session_state:
        st.session_state["conversation_state"] = ConversationState.create()


def _conversation_state() -> ConversationState:
    return cast(ConversationState, st.session_state["conversation_state"])


def _rename_active_conversation(title: str) -> None:
    state = _conversation_state()
    state.rename(state.active_id, title)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand-mark">t</div>', unsafe_allow_html=True)
        st.markdown("### Tiny Agent")
        st.caption("A compact, observable agent runtime")
        st.markdown("---")

        with st.form("connection_settings"):
            api_url = st.text_input(
                "API endpoint",
                value=cast(str, st.session_state["active_api_url"]),
                help="The address of the running FastAPI service.",
            )
            user_id = st.text_input(
                "User ID",
                value=cast(str, st.session_state["active_user_id"]),
                help="Owns this conversation window and shared todo state.",
            )
            applied = st.form_submit_button(
                "Apply settings", icon=":material/check:", width="stretch"
            )

        if applied:
            try:
                TinyAgentClient(api_url)
            except ValueError as exc:
                st.error(str(exc))
            else:
                if not user_id.strip():
                    st.error("User ID cannot be blank.")
                else:
                    st.session_state["active_api_url"] = api_url.strip().rstrip("/")
                    st.session_state["active_user_id"] = user_id.strip()
                    _conversation_state().reset()
                    st.rerun()

        state = _conversation_state()
        if st.button(
            "New conversation",
            icon=":material/add_comment:",
            type="primary",
            width="stretch",
        ):
            state.new_conversation()
            st.rerun()

        st.markdown("#### Conversations")
        st.caption(f"{len(state.conversations)} saved in this browser tab")
        for conversation in state.conversations:
            is_active = conversation.id == state.active_id
            if st.button(
                conversation.title,
                key=f"select-conversation-{conversation.id}",
                icon=":material/chat_bubble:" if is_active else ":material/chat_bubble_outline:",
                type="primary" if is_active else "tertiary",
                width="stretch",
            ):
                state.select(conversation.id)
                st.rerun()

        active = state.active
        with st.expander("Manage active conversation", icon=":material/settings:"):
            with st.form(f"rename-conversation-{active.id}", border=False):
                new_title = st.text_input(
                    "Conversation name",
                    value=active.title,
                    max_chars=80,
                    key=f"conversation-title-{active.id}-{active.title_revision}",
                )
                rename = st.form_submit_button(
                    "Rename", icon=":material/edit:", width="stretch"
                )
            if rename:
                try:
                    _rename_active_conversation(new_title)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()

            if active.api_session_id:
                st.caption("Runtime session ID")
                st.code(active.api_session_id, language=None)
            else:
                st.caption("A runtime session is created with the first message.")

            if st.button(
                "Remove conversation",
                key=f"remove-conversation-{active.id}",
                icon=":material/delete_outline:",
                type="tertiary",
                width="stretch",
                help="Removes this conversation from the frontend.",
            ):
                state.remove(active.id)
                st.rerun()

        st.markdown("#### Available tools")
        st.caption("Calculator · Search · Todo · Weather")


def _render_trace(trace: tuple[TraceEntry, ...]) -> None:
    if not trace:
        return
    tool_names: list[str] = []
    for event in trace:
        tool_name = event.data.get("name")
        if event.kind == "tool_started" and isinstance(tool_name, str):
            tool_names.append(tool_name)
    summary = " · ".join(tool_names) if tool_names else "Direct answer"
    with st.expander(f"Run trace · {summary}"):
        for event in trace:
            label = event.kind.replace("_", " ").title()
            st.markdown(f"**{label}** &nbsp; `#{event.sequence}`")
            if event.data:
                st.json(event.data, expanded=False)


def _render_message(message: ChatMessage) -> None:
    with st.chat_message(message.role):
        if message.is_error:
            st.error(message.content)
        else:
            st.markdown(message.content)
        _render_trace(message.trace)


def _render_empty_state() -> str | None:
    empty_state = st.empty()
    selected: str | None = None
    with empty_state.container():
        st.markdown(
            """
            <div class="welcome-card">
              <div class="eyebrow">READY WHEN YOU ARE</div>
              <h2>What can Tiny Agent do?</h2>
              <p>Ask a question, use a built-in tool, or continue a thought across several
              turns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        columns = st.columns(3)
        for column, (label, prompt) in zip(columns, SUGGESTIONS, strict=True):
            with column:
                if st.button(
                    f"{label}\n\n{prompt}",
                    key=f"suggestion-{label}",
                    width="stretch",
                ):
                    selected = prompt
    if selected:
        empty_state.empty()
    return selected


def _submit(prompt: str) -> None:
    conversation = _conversation_state().active
    messages = conversation.messages
    conversation.title_from_prompt(prompt)
    user_message = ChatMessage(role="user", content=prompt)
    messages.append(user_message)
    _render_message(user_message)

    api_url = cast(str, st.session_state["active_api_url"])
    user_id = cast(str, st.session_state["active_user_id"])
    session_id = conversation.api_session_id

    activity = st.empty()
    try:
        client = TinyAgentClient(api_url)
        with activity.container(), st.status(
            "Tiny Agent is working…", expanded=False
        ) as status:
            if session_id is None:
                session_id = client.create_session(user_id)
                conversation.api_session_id = session_id
            result = client.send_message(user_id, session_id, prompt)
            status.update(label="Run complete", state="complete")
        if result.status == "completed" and result.answer:
            assistant_message = ChatMessage(
                role="assistant", content=result.answer, trace=result.trace
            )
        else:
            error_message = (
                result.error.message
                if result.error
                else "The run ended without an answer."
            )
            assistant_message = ChatMessage(
                role="assistant",
                content=error_message,
                trace=result.trace,
                is_error=True,
            )
    except (FrontendAPIError, ValueError) as exc:
        safe_message = exc.safe_message if isinstance(exc, FrontendAPIError) else str(exc)
        assistant_message = ChatMessage(
            role="assistant", content=safe_message, is_error=True
        )

    activity.empty()
    messages.append(assistant_message)
    st.rerun()


def _styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 82% 8%, rgba(113, 79, 255, .14), transparent 28rem),
                radial-gradient(circle at 18% 88%, rgba(48, 203, 180, .08), transparent 24rem),
                #090d18;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: rgba(13, 19, 34, .92);
            border-right: 1px solid rgba(149, 163, 192, .12);
        }
        .block-container { max-width: 880px; padding-top: 3.2rem; padding-bottom: 7rem; }
        .brand-mark {
            display: grid; place-items: center; width: 42px; height: 42px;
            border-radius: 13px; margin-bottom: .8rem; color: white;
            font-size: 1.45rem; font-weight: 750; font-style: italic;
            background: linear-gradient(145deg, #8b6cff, #5a3ee4);
            box-shadow: 0 12px 30px rgba(112, 79, 255, .28);
        }
        .hero { margin-bottom: 2.2rem; }
        .eyebrow {
            color: #9c8cff; font-size: .72rem; font-weight: 750;
            letter-spacing: .16em; margin-bottom: .55rem;
        }
        .hero h1 {
            font-size: clamp(2.65rem, 7vw, 4.9rem); line-height: .98;
            letter-spacing: -.055em; margin: 0 0 1rem; color: #f5f7ff;
        }
        .hero p { color: #9aa7bf; font-size: 1.08rem; max-width: 610px; line-height: 1.65; }
        .welcome-card {
            padding: 1.6rem 1.7rem; margin: 1rem 0 1rem;
            border: 1px solid rgba(153, 137, 255, .20); border-radius: 22px;
            background: linear-gradient(145deg, rgba(126, 96, 255, .09), rgba(18, 27, 48, .70));
        }
        .welcome-card h2 { margin: .15rem 0 .45rem; color: #eef1ff; letter-spacing: -.02em; }
        .welcome-card p { margin: 0; color: #96a3bb; }
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(151, 164, 190, .11); border-radius: 18px;
            background: rgba(18, 26, 45, .65); padding: .35rem .7rem;
        }
        [data-testid="stChatInput"] { border-color: rgba(139, 108, 255, .45); }
        .stButton > button, .stFormSubmitButton > button { border-radius: 12px; }
        [data-testid="stExpander"] { border-color: rgba(151, 164, 190, .14); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the complete Streamlit application."""
    st.set_page_config(
        page_title="Tiny Agent",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_state()
    _styles()
    _render_sidebar()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">TINY RUNTIME · CLEAR DECISIONS</div>
          <h1>Think small.<br>Do useful things.</h1>
          <p>A direct window into Tiny Agent. Every answer stays bounded, every tool call is
          validated, and every run leaves a readable trace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    messages = _conversation_state().active.messages
    selected_prompt = _render_empty_state() if not messages else None
    for message in messages:
        _render_message(message)

    typed_prompt = st.chat_input("Ask Tiny Agent anything…", max_chars=100_000)
    prompt = selected_prompt or typed_prompt
    if prompt:
        _submit(prompt.strip())


if __name__ == "__main__":
    main()
