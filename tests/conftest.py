"""Shared pytest settings.

Tests that call outside services (FIRMS, gridMET, terrain tiles) are marked
@pytest.mark.network and only run with:  python -m pytest --network
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", help="also run tests that call outside services")


def pytest_configure(config):
    config.addinivalue_line("markers", "network: calls an outside service; runs only with --network")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="calls an outside service; run with --network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
