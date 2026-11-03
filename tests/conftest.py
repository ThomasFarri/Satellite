from pathlib import Path

import pytest

from opticalpass.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def example_config():
    return load_config(ROOT / "configs" / "example.yaml")
