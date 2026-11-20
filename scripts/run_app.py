#!/usr/bin/env python3
"""Start the whole system with one command: the API, then the Streamlit app once the API is up.

    python scripts/run_app.py                      # API on :8000, app on :8501
    python scripts/run_app.py --api-port 8010 --ui-port 8510

Ctrl+C stops both. Uses the current Python (run it from the activated virtual environment).
"""
from __future__ import annotations

import argparse
import json
import os
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


def main():
    p = argparse.ArgumentParser(description="Run the Food Pairing API and Streamlit app")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--ui-port", type=int, default=8501)
    args = p.parse_args()

    api_url = f"http://127.0.0.1:{args.api_port}"
    env = os.environ | {"FOODPAIRING_API_URL": api_url}
    procs = []
    try:
        print(f"Starting the API on {api_url} (loading the model and the recipes)...")
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app",
                                "--host", "127.0.0.1", "--port", str(args.api_port)], cwd=ROOT, env=env)
        procs.append(api)
        health = wait_for_api(api_url, api)
        model = health.get("model")
        print(f"API ready: model = {model}"
              + ("" if model == "nlgcl" else "  (MOCK: artifacts/ingredient_embeddings.npy missing?)"))

        print(f"Starting the app on http://localhost:{args.ui_port} ...")
        procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                                       "--server.port", str(args.ui_port), "--server.headless", "true"],
                                      cwd=ROOT, env=env))
        print(f"\n  Open http://localhost:{args.ui_port}  (demo account: demo / demo1234)"
              "\n  Ctrl+C to stop.\n")
        while all(pr.poll() is None for pr in procs):
            time.sleep(1)
        raise SystemExit("A process stopped; see the error above.")
    except KeyboardInterrupt:
        print("\nStopping...")
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
