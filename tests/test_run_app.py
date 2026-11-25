"""scripts/run_app.py --lan helpers: the advertised IP and the QR code of the showcase URL."""
import importlib.util
import ipaddress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_app", ROOT / "scripts" / "run_app.py")
run_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_app)


def test_lan_ip_is_an_ipv4_address():
    assert ipaddress.ip_address(run_app.lan_ip()).version == 4


def test_show_lan_access_writes_the_qr_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_app, "ROOT", tmp_path)
    run_app.show_lan_access("http://192.168.1.10:8501")
    out = capsys.readouterr().out
    assert "http://192.168.1.10:8501" in out
    assert (tmp_path / "artifacts" / "showcase_qr.png").stat().st_size > 0
