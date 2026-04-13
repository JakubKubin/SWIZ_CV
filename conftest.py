# conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--visualize",
        action="store_true",
        default=True,
        help="Zapisz obrazy debugowe do ścieżki",
    )


@pytest.fixture
def visualize(request):
    return request.config.getoption("--visualize")
