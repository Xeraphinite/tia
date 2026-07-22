"""Shared test-suite classification checks."""

from __future__ import annotations

import pytest

PRIMARY_TEST_CATEGORIES = frozenset(
    {"unit", "integration", "environment", "provider", "e2e"}
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Require every test to belong to exactly one primary category."""
    invalid: list[str] = []
    for item in items:
        categories = sorted(
            name
            for name in PRIMARY_TEST_CATEGORIES
            if item.get_closest_marker(name) is not None
        )
        if len(categories) != 1:
            rendered = ", ".join(categories) if categories else "none"
            invalid.append(f"{item.nodeid} ({rendered})")
    if invalid:
        details = "\n".join(f"- {entry}" for entry in invalid)
        raise pytest.UsageError(
            "tests must have exactly one primary category marker:\n" + details
        )
