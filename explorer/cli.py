import subprocess
import sys
import os


def run():
    dashboard = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ptycho_dashboard.py")
    )
    print("=" * 52)
    print("  Ptycho Dashboard")
    print("  http://localhost:5006/ptycho_dashboard")
    print("  Press Ctrl+C to stop.")
    print("=" * 52)
    subprocess.run(
        [
            sys.executable, "-m", "panel", "serve", dashboard,
            "--port", "5006",
            "--allow-websocket-origin", "localhost:5006",
        ]
    )
