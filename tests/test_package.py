import importlib

import pytest

pytestmark = pytest.mark.phase1

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


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"higgs4l.{name}")


def test_config_loads(cfg):
    assert cfg["release"] == "2025e-13tev-beta"
    assert cfg["skim"] == "exactly4lep"
    assert cfg["lumi_fb"] == 36.6
