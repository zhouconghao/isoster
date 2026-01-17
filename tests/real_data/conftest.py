"""
Pytest configuration for real data tests.

These tests use real astronomical data and are skipped by default in CI.
To run them: pytest tests/real_data/ -m real_data
"""

import pytest


def pytest_collection_modifyitems(items):
    """Mark all tests in this directory as real_data."""
    for item in items:
        if "real_data" in str(item.fspath):
            item.add_marker(pytest.mark.real_data)


# Skip real_data tests by default unless explicitly requested
def pytest_configure(config):
    """Skip real_data tests unless -m real_data is specified."""
    if not config.option.markexpr:
        # Default: skip real_data tests
        config.option.markexpr = "not real_data"
