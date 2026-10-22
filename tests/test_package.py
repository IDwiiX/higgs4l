import importlib
from pathlib import Path

import pytest
import yaml

MODULES = [
    "data",
    "selection",
    "weights",
    "kinematics",
    "massfit",
    "stats",
    "ml",
    "plotting",
]
CONFIG = Path(__file__).resolve().parent.parent / "config" / "analysis.yaml"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"higgs4l.{name}")


def test_config_loads():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["release"] == "2025e-13tev-beta"
    assert config["skim"] == "exactly4lep"
    assert config["lumi_fb"] == 36.6
