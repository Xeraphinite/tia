import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from litellm import completion

MODEL = "openrouter/openai/gpt-5.4-nano"

load_dotenv(Path(".env"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_LLM_TESTS") != "1",
        reason="set RUN_LIVE_LLM_TESTS=1 to call OpenRouter",
    ),
]


def test_openrouter_completion() -> None:
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"]
    failure: str | None = None
    response = None

    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly TIA_OK"}],
            max_tokens=32,
            timeout=30,
            max_retries=2,
        )
    except Exception as exc:
        # LiteLLM exceptions may contain request headers, including credentials.
        failure = type(exc).__name__

    if failure is not None:
        pytest.fail(f"OpenRouter smoke test failed: {failure}", pytrace=False)

    assert response is not None
    assert response.choices[0].message.content == "TIA_OK"
