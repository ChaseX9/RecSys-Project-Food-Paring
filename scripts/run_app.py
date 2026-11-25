#!/usr/bin/env python3
"""Start the whole system with one command: the API, then the Streamlit app once the API is up.

    python scripts/run_app.py                      # API on :8000, app on :8501
    python scripts/run_app.py --api-port 8010 --ui-port 8510
    python scripts/run_app.py --lan                # showcase: app reachable from the same Wi-Fi

With --lan, only the Streamlit app listens on the network (0.0.0.0); the API stays on 127.0.0.1
because it has no authentication of its own. The script prints http://<this laptop's IP>:<ui-port>
and saves a QR code to artifacts/showcase_qr.png for visitors' phones.

Ctrl+C stops both. Uses the current Python (run it from the activated virtual environment).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_for_api(url: str, proc: subprocess.Popen, timeout: float = 120) -> dict:
    """Poll /health until the API answers (it loads the model and the recipes first)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"The API stopped during startup (exit code {proc.returncode}); see the error above.")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                return json.load(r)
        except OSError:
            time.sleep(1)
    raise SystemExit(f"The API did not answer on {url} within {timeout:.0f} s.")


def port_in_use(port: int) -> bool:
    """True if something already listens on this port, on IPv4 *or* IPv6.

    Windows lets two programs listen on the same port when one binds IPv6 (::) and the other
    IPv4 (0.0.0.0): uvicorn/streamlit start without error, and visitors may land on the other
    program. So we connect instead of relying on a bind error."""
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((host, port)) == 0:
                    return True
        except OSError:  # no IPv6 on this machine
            pass
    return False


def free_port(port: int, name: str, tries: int = 50) -> int:
    """``port`` if free, else the next free one (and say so)."""
    for p in range(port, port + tries):
        if not port_in_use(p):
            if p != port:
                print(f"Port {port} is already used by another program: the {name} uses port {p} instead.",
                      flush=True)
            return p
    raise SystemExit(f"No free port for the {name} between {port} and {port + tries - 1}.")


def lan_ip() -> str:
    """IP of the interface that reaches the network (no packet is sent: UDP connect only picks a route)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def show_lan_access(url: str) -> None:
    """Print the URL to give visitors and save it as a QR code (artifacts/showcase_qr.png)."""
    print(f"\n  Showcase URL (same Wi-Fi): {url}", flush=True)
    if url.split("//")[1].startswith("127."):
        print("  WARNING: no network found (127.x). Connect to the Wi-Fi or a phone hotspot first.", flush=True)
    try:
        import segno
    except ImportError:
        print("  (pip install segno to also get a QR code)", flush=True)
        return
    qr = segno.make(url, error="m")
    path = ROOT / "artifacts" / "showcase_qr.png"
    path.parent.mkdir(exist_ok=True)
    qr.save(path, scale=12, border=2)
    print(f"  QR code saved to {path.relative_to(ROOT)} (show it on the stand)", flush=True)
    if sys.platform == "win32":
        print("  If other devices cannot connect: allow Python in the Windows firewall popup (tick both\n"
              "  Private and Public), or as admin: netsh advfirewall firewall add rule name=FoodPairing"
              f" dir=in action=allow protocol=TCP localport={url.rsplit(':', 1)[1]}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Run the Food Pairing API and Streamlit app")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--ui-port", type=int, default=8501)
    p.add_argument("--lan", action="store_true",
                   help="make the app reachable from other devices on the same network (showcase)")
    p.add_argument("--ip", help="with --lan: IP to advertise, if the detected one is wrong (VPN, several networks)")
    args = p.parse_args()
    args.api_port = free_port(args.api_port, "API")
    args.ui_port = free_port(args.ui_port, "app")

    api_url = f"http://127.0.0.1:{args.api_port}"
    env = os.environ | {"FOODPAIRING_API_URL": api_url}
    procs = []
    try:
        print(f"Starting the API on {api_url} (loading the model and the recipes)...", flush=True)
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app",
                                "--host", "127.0.0.1", "--port", str(args.api_port)], cwd=ROOT, env=env)
        procs.append(api)
        health = wait_for_api(api_url, api)
        model = health.get("model")
        print(f"API ready: model = {model}"
              + ("" if model == "nlgcl" else "  (MOCK: artifacts/ingredient_embeddings.npy missing?)"),
              flush=True)

        print(f"Starting the app on http://localhost:{args.ui_port} ...", flush=True)
        ui_cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                  "--server.port", str(args.ui_port), "--server.headless", "true"]
        if args.lan:
            ui_cmd += ["--server.address", "0.0.0.0"]
        procs.append(subprocess.Popen(ui_cmd, cwd=ROOT, env=env))
        print(f"\n  Open http://localhost:{args.ui_port}  (demo account: demo / demo1234)", flush=True)
        if args.lan:
            show_lan_access(f"http://{args.ip or lan_ip()}:{args.ui_port}")
        print("\n  Ctrl+C to stop.\n", flush=True)
        while all(pr.poll() is None for pr in procs):
            time.sleep(1)
        raise SystemExit("A process stopped; see the error above.")
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()


if __name__ == "__main__":
    main()
