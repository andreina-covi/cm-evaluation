from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tiny_items_json() -> Path:
    return FIXTURES / "items_house_tiny.json"
