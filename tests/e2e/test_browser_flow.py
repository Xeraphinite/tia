"""Browser-level acceptance flow across Streamlit, HTTP, tools, and SQLite."""

from __future__ import annotations

import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, expect, sync_playwright

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("TIA_RUN_E2E_TESTS") != "1",
        reason="browser end-to-end tests are opt-in",
    ),
]

ROOT = Path(__file__).parents[2]
WEATHER_PROMPT = (
    "Use get_weather for Shanghai, then use todo to add 'bring an umbrella'. "
    "Report both results."
)
WEEKLY_PROMPT = (
    "Draft a short weekly report, then use todo to add 'send weekly report'. "
    "Confirm the saved item."
)
LIST_PROMPT = (
    "Use todo to list only this window's items. Also mention this conversation's topic."
)
RESTART_PROMPT = "Use todo to list this window's items after the API restart."


@dataclass(slots=True)
class ServiceProcess:
    """A child service with diagnostics retained in a temporary log."""

    process: subprocess.Popen[bytes]
    log_path: Path

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def log(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def test_two_windows_tools_followups_and_restart_are_isolated(tmp_path: Path) -> None:
    """Exercise the complete user-visible flow and durable session recovery."""
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required for the real-provider E2E test.")
    api_port = _free_port()
    frontend_port = _free_port()
    database_path = tmp_path / "e2e.sqlite3"
    api = _start_api(api_port, database_path, tmp_path / "api.log")
    frontend = _start_frontend(
        frontend_port,
        api_port,
        tmp_path / "frontend.log",
    )

    try:
        _wait_for_url(f"http://127.0.0.1:{api_port}/openapi.json", api)
        _wait_for_url(f"http://127.0.0.1:{frontend_port}", frontend)
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                pytest.fail(
                    "Chromium is unavailable; run `uv run playwright install chromium`. "
                    f"Original error: {exc}"
                )
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.goto(
                    f"http://127.0.0.1:{frontend_port}",
                    wait_until="domcontentloaded",
                )
                weather_answer = _submit(page, WEATHER_PROMPT)
                assert "shanghai" in weather_answer.casefold()
                assert "umbrella" in weather_answer.casefold()
                weather_trace = page.get_by_text(
                    re.compile(r"Run trace.*get_weather.*todo", re.IGNORECASE)
                ).last
                expect(weather_trace).to_be_visible()

                page.get_by_role("button", name="New conversation").click()
                expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(
                    0, timeout=10_000
                )
                weekly_answer = _submit(page, WEEKLY_PROMPT)
                assert "weekly report" in weekly_answer.casefold()
                expect(page.get_by_text("Run trace · todo", exact=True).last).to_be_visible()

                _select_conversation(page, WEATHER_PROMPT)
                weather_followup = _submit(page, LIST_PROMPT).casefold()
                assert "umbrella" in weather_followup
                assert "send weekly report" not in weather_followup

                _select_conversation(page, WEEKLY_PROMPT)
                weekly_followup = _submit(page, LIST_PROMPT).casefold()
                assert "send weekly report" in weekly_followup
                assert "umbrella" not in weekly_followup

                api.stop()
                api = _start_api(api_port, database_path, tmp_path / "api-restarted.log")
                _wait_for_url(f"http://127.0.0.1:{api_port}/openapi.json", api)

                _select_conversation(page, WEATHER_PROMPT)
                prior_message_count = page.locator('[data-testid="stChatMessage"]').count()
                restarted_answer = _submit(page, RESTART_PROMPT).casefold()
                expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(
                    prior_message_count + 2
                )
                assert "umbrella" in restarted_answer
                assert "send weekly report" not in restarted_answer
            finally:
                browser.close()

        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                "SELECT session_id, title FROM todos ORDER BY title"
            ).fetchall()
        assert {str(row[1]) for row in rows} == {
            "bring an umbrella",
            "send weekly report",
        }
        assert len({str(row[0]) for row in rows}) == 2
    finally:
        frontend.stop()
        api.stop()


def _submit(page: Page, prompt: str) -> str:
    messages = page.locator('[data-testid="stChatMessage"]')
    prior_count = messages.count()
    chat_input = page.get_by_placeholder("Ask Tiny Agent anything…")
    expect(chat_input).to_be_visible(timeout=15_000)
    chat_input.fill(prompt)
    chat_input.press("Enter")
    expect(messages).to_have_count(prior_count + 2, timeout=70_000)
    return messages.last.inner_text()


def _select_conversation(page: Page, title: str) -> None:
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.locator("button").filter(has_text=_conversation_title(title)).click()
    first_message = page.locator('[data-testid="stChatMessage"]').first
    expect(first_message).to_contain_text(title, timeout=10_000)


def _conversation_title(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    return f"{normalized[:45].rstrip()}…" if len(normalized) > 45 else normalized


def _start_api(port: int, database_path: Path, log_path: Path) -> ServiceProcess:
    environment = os.environ.copy()
    environment["TIA_E2E_DATABASE_PATH"] = str(database_path)
    return _start_process(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.e2e.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        environment,
        log_path,
    )


def _start_frontend(port: int, api_port: int, log_path: Path) -> ServiceProcess:
    environment = os.environ.copy()
    environment["TIA_API_URL"] = f"http://127.0.0.1:{api_port}"
    environment["TIA_USER_ID"] = "e2e-user"
    return _start_process(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "streamlit_app.py"),
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--server.headless=true",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
        ],
        environment,
        log_path,
    )


def _start_process(
    command: list[str], environment: dict[str, str], log_path: Path
) -> ServiceProcess:
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return ServiceProcess(process=process, log_path=log_path)


def _wait_for_url(url: str, service: ServiceProcess, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if service.process.poll() is not None:
            pytest.fail(f"service exited before becoming ready:\n{service.log()}")
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status < 500:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.1)
    pytest.fail(f"service did not become ready at {url}:\n{service.log()}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])
