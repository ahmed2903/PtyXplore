import subprocess
import sys
import os
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--allow-websocket-origin", dest="ws_origin", default=None,
                        help="Extra hostname:port to allow (for tunnels). Use '*' to allow all.")
    args = parser.parse_args()

    dashboard = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ptycho_dashboard.py")
    )
    print("=" * 52)
    print("  Ptycho Dashboard")
    print(f"  http://localhost:{args.port}/ptycho_dashboard")
    print("  Press Ctrl+C to stop.")
    print("=" * 52)

    cmd = [
        sys.executable, "-m", "panel", "serve", dashboard,
        "--port", str(args.port),
        "--allow-websocket-origin", f"localhost:{args.port}",
    ]
    if args.ws_origin:
        cmd += ["--allow-websocket-origin", args.ws_origin]

    subprocess.run(cmd)
