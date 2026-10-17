"""Entry point shared by python start.py and python -m local_chat."""

import argparse
import signal
import sys
import threading
import webbrowser

from .config import Settings
from .engine import Engine
from .server import Application


def main():
    parser = argparse.ArgumentParser(
        description="Local Chat with GGUF models and DuckDuckGo search."
    )
    parser.add_argument("--port", type=int, default=8080, help="interface port (default: 8080)")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the browser automatically"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    engine = Engine(Settings())
    try:
        app = Application(("127.0.0.1", args.port), engine)
    except OSError:
        print(
            f"Port {args.port} is unavailable. Close the other application or use python start.py --port 8081.",
            file=sys.stderr,
        )
        return 1
    url = f"http://127.0.0.1:{app.server_port}"
    print(f"\n  Local Chat\n  {url}\n\n  Ctrl+C to quit and release the model.\n", flush=True)

    def shutdown(*_):
        app.cancel()
        threading.Thread(target=app.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if not args.no_browser:

        def open_browser():
            try:
                if not webbrowser.open(url):
                    print("Open the link above in your browser.", flush=True)
            except webbrowser.Error:
                print("Open the link above in your browser.", flush=True)

        threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=engine.restore, daemon=True).start()
    try:
        app.serve_forever(poll_interval=0.2)
    finally:
        app.cancel()
        app.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
