"""The showcase demo scenario (scripts/demo_check.py) still shows every effect on the served model."""
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("numpy")
if not (ROOT / "artifacts" / "ingredient_embeddings.npy").exists():
    pytest.skip("no trained model", allow_module_level=True)


def _demo_module():
    spec = importlib.util.spec_from_file_location("demo_check", ROOT / "scripts" / "demo_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_scenario_passes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FOODPAIRING_SEED_DEMO", "0")
    monkeypatch.delenv("FOODPAIRING_MOCK", raising=False)
    from api.main import create_app

    demo = _demo_module()
    with TestClient(create_app(str(tmp_path / "demo.db"))) as c:
        d = demo.Demo(c)
        demo.run(d)
    assert d.failures == [], capsys.readouterr().out
