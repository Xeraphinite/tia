from importlib.metadata import version


def test_runtime_dependencies_are_installed() -> None:
    assert version("fastapi")
    assert version("litellm").startswith("1.75.")
    assert version("pydantic").startswith("2.")
    assert version("python-dotenv")
    assert version("uvicorn")
