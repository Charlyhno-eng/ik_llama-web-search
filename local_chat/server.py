"""Loopback-only HTTP app with per-launch API token and no external frontend assets."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit

from .chat import Job, generate, validate_messages
from .config import AppError, browse
from .search import safe_url

STATIC = Path(__file__).parent / "static"


class Application(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine):
        self.engine = engine
        self.token = secrets.token_urlsafe(32)
        self.job = None
        self.job_lock = threading.Lock()
        super().__init__(address, Handler)

    def cancel(self, identifier=None):
        with self.job_lock:
            if self.job and (identifier is None or identifier == self.job.id):
                self.job.cancel()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Prompts and local model paths must not land in HTTP access logs.
        pass

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        super().end_headers()

    def check_request(self):
        port = self.server.server_port
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            raise AppError("Host not allowed.", 403)
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            raise AppError("Origin not allowed.", 403)
        if urlsplit(self.path).path.startswith("/api/"):
            token = self.headers.get("X-Local-Chat-Token", "")
            if not secrets.compare_digest(token, self.server.token):
                raise AppError("Session expired. Reload this page.", 403)

    def send_bytes(self, body, kind="application/json; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), status=status)

    def read_json(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise AppError("Invalid request format.", 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise AppError("Request is empty or too large.", 413)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("object required")
            return data
        except (ValueError, UnicodeError) as exc:
            raise AppError("Invalid JSON.") from exc

    def do_GET(self):
        try:
            self.check_request()
            path = urlsplit(self.path).path
            if path == "/api/status":
                self.send_json(self.server.engine.status())
            elif path == "/api/files":
                self.send_json(
                    browse(parse_qs(urlsplit(self.path).query).get("directory", [None])[0])
                )
            elif path == "/api/logs":
                self.send_json({"text": self.server.engine.logs()})
            elif path in ("/", "/app.js", "/style.css", "/icon.svg"):
                name = "index.html" if path == "/" else path[1:]
                body = (STATIC / name).read_bytes()
                if name == "index.html":
                    body = body.replace(b"__SESSION_TOKEN__", self.server.token.encode())
                kind = {
                    "index.html": "text/html; charset=utf-8",
                    "app.js": "text/javascript; charset=utf-8",
                    "style.css": "text/css; charset=utf-8",
                    "icon.svg": "image/svg+xml",
                }[name]
                self.send_bytes(body, kind)
            else:
                raise AppError("Page not found.", 404)
        except AppError as exc:
            self.close_connection = True
            self.send_json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError:
            self.close_connection = True
            self.send_json({"error": "Unable to read this local file."}, 500)

    def do_POST(self):
        streaming = False
        job = None
        acquired = False
        try:
            self.check_request()
            data = self.read_json()
            path = urlsplit(self.path).path
            engine = self.server.engine
            if path == "/api/settings":
                engine.load(data)
                self.send_json(engine.status(), 202)
            elif path == "/api/cancel":
                self.server.cancel(data.get("id", ""))
                self.send_json({"ok": True})
            elif path == "/api/chat":
                messages = validate_messages(data.get("messages"))
                previous_sources = data.get("previous_sources", [])
                if (
                    not isinstance(previous_sources, list)
                    or len(previous_sources) > 5
                    or any(
                        not isinstance(item, dict)
                        or not safe_url(item.get("url"))
                        or not isinstance(item.get("title", ""), str)
                        or len(item.get("title", "")) > 240
                        for item in previous_sources
                    )
                ):
                    raise AppError("Invalid previous-source hints.")
                if type(data.get("web")) is not bool:
                    raise AppError("Specify whether web search is enabled.")
                identifier = data.get("id")
                if not isinstance(identifier, str) or not 1 <= len(identifier) <= 100:
                    raise AppError("Invalid request ID.")
                if not engine.operation.acquire(blocking=False):
                    raise AppError("Another operation is in progress.", 409)
                acquired = True
                if engine.status()["state"] != "ready":
                    raise AppError("Load a model in Settings first.", 409)
                job = Job(identifier)
                with self.server.job_lock:
                    self.server.job = job
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                streaming = True
                generate(engine, messages, data["web"], job, self.emit, previous_sources)
            else:
                raise AppError("Action not found.", 404)
        except AppError as exc:
            if streaming:
                try:
                    self.emit("cancelled" if exc.status == 499 else "error", {"message": str(exc)})
                except OSError:
                    pass
            else:
                self.close_connection = True
                self.send_json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            if job:
                job.cancel()
        except Exception:
            if streaming:
                try:
                    self.emit("error", {"message": "An unexpected error occurred. Please retry."})
                except OSError:
                    pass
            else:
                self.close_connection = True
                self.send_json({"error": "A local error occurred. Check file permissions."}, 500)
        finally:
            if job:
                with self.server.job_lock:
                    if self.server.job is job:
                        self.server.job = None
            if acquired:
                self.server.engine.operation.release()

    def emit(self, event, data):
        self.wfile.write(
            (
                "event: " + event + "\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n"
            ).encode()
        )
        self.wfile.flush()
