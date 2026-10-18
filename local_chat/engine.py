"""Own the external llama-server lifecycle; no inference implementation here."""

from collections import deque
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .config import AppError, ROOT, gguf_path

# Loopback requests must not be redirected through an HTTP_PROXY from the environment.
LOCAL_HTTP = build_opener(ProxyHandler({}))


def process_options():
    """Isolate owned children from the terminal's Ctrl+C on either platform."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.operation = threading.Lock()
        self.closing = threading.Event()
        self.process = None
        self.builder = None
        self.port = None
        self.state = "idle"
        self.message = "Choose a model in Settings."
        self.active = None
        self.log_path = settings.directory / "llama-server.log"
        self.build_log_path = settings.directory / "build.log"

    def status(self):
        if self.state == "ready" and self.process and self.process.poll() is not None:
            self.state, self.message = "error", "The engine stopped. Reload your model in Settings."
            self.active = None
        return {
            "state": self.state,
            "message": self.message,
            "settings": self.settings.value.copy(),
            "active": self.active,
            "warning": self.settings.warning,
        }

    def executable(self):
        override = os.environ.get("LLAMA_SERVER")
        if override:
            path = Path(override).expanduser().resolve()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise AppError("LLAMA_SERVER does not point to an accessible executable.")
            return str(path)
        name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        for path in (ROOT / "build/bin" / name, ROOT / "build/bin/Release" / name):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        return None

    def ensure_executable(self):
        executable = self.executable()
        if executable:
            return executable
        # CMake can discover Visual Studio even when cl.exe is not on PATH.
        if not shutil.which("cmake") or (
            sys.platform != "win32" and not (shutil.which("c++") or shutil.which("clang++"))
        ):
            raise AppError(
                "First run: install CMake and a C++20 compiler, then reload the model. "
                "Use build-essential on Ubuntu/Debian, Xcode Command Line Tools on macOS, "
                "or Visual Studio Build Tools with Desktop development with C++ on Windows."
            )
        self.state = "building"
        self.message = "First run: building the CPU server. This may take several minutes."
        commands = [
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(ROOT / "build"),
                "-DGGML_NATIVE=ON",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            [
                "cmake",
                "--build",
                str(ROOT / "build"),
                "--target",
                "llama-server",
                "--config",
                "Release",
                "-j",
                "2",
            ],
        ]
        with self.build_log_path.open("w", encoding="utf-8") as log:
            for command in commands:
                if self.closing.is_set():
                    raise AppError("Application stopped.")
                self.builder = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, **process_options()
                )
                while self.builder.poll() is None:
                    if self.closing.wait(0.2):
                        self._stop_process(self.builder)
                        raise AppError("Build interrupted because the application closed.")
                if self.builder.returncode != 0:
                    raise AppError(
                        "Build failed. Check .local-chat/build.log, then reload the model."
                    )
                self.builder = None
        executable = self.executable()
        if not executable:
            raise AppError("The build did not produce llama-server.")
        return executable

    def load(self, data):
        value = self.settings.validate(data)
        if not self.operation.acquire(blocking=False):
            raise AppError(
                "An operation is in progress. Stop the response or wait for loading to finish.", 409
            )
        try:
            self.settings.save(value)
        except Exception:
            self.operation.release()
            raise
        self.state, self.message = "loading", "Preparing the model…"
        threading.Thread(target=self._load, args=(value,), daemon=True).start()

    def restore(self):
        if self.settings.value["model_path"]:
            try:
                self.load(self.settings.value.copy())
            except (AppError, OSError) as exc:
                self.state, self.message = "error", str(exc)

    def _load(self, value):
        try:
            self._stop_process(self.process)
            self.process, self.active = None, None
            executable = self.ensure_executable()
            gguf_path(value["model_path"])
            if self.closing.is_set():
                return
            with socket.socket() as port_socket:
                port_socket.bind(("127.0.0.1", 0))
                self.port = port_socket.getsockname()[1]
            command = [
                executable,
                "--model",
                value["model_path"],
                "--ctx-size",
                str(value["context_size"]),
                "--threads",
                str(value["threads"]),
                "--threads-batch",
                str(value.get("threads_batch") or value["threads"]),
                "--jinja",
                "--reasoning-tokens",
                "none",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--parallel",
                "1",
                "--no-context-shift",
            ]
            self.state, self.message = "loading", "Loading " + Path(value["model_path"]).name + "…"
            with self.log_path.open("w", encoding="utf-8") as log:
                self.process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, **process_options()
                )
            deadline = time.monotonic() + 600
            while not self.closing.is_set() and time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise AppError(
                        "The model could not be loaded. Check compatibility and available memory. Details: .local-chat/llama-server.log."
                    )
                try:
                    self.call("/health", timeout=1)
                except (AppError, OSError):
                    self.closing.wait(0.3)
                    continue
                self.active = value.copy()
                self.state, self.message = "ready", "Ready"
                return
            raise AppError("Model loading was interrupted or exceeded ten minutes.")
        except (AppError, OSError) as exc:
            self._stop_process(self.process)
            self.process, self.active = None, None
            self.state, self.message = "error", str(exc)
        finally:
            self.operation.release()

    def call(self, path, data=None, timeout=30):
        request = Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with LOCAL_HTTP.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise AppError(
                "The engine rejected the request. Check the model and its chat template.", 502
            ) from exc
        except (URLError, OSError, ValueError) as exc:
            raise AppError("The local engine is unreachable. Reload the model.", 503) from exc

    @staticmethod
    def _stop_process(process):
        if not process or process.poll() is not None:
            return
        if sys.platform == "win32":
            # Terminate the whole owned tree, including CMake's compiler children.
            # Windows has no killpg; terminating only CMake would leave them running.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def close(self):
        self.closing.set()
        self._stop_process(self.builder)
        self._stop_process(self.process)
        # A load worker may have passed its closing check just before close().
        with self.operation:
            self._stop_process(self.builder)
            self._stop_process(self.process)

    def logs(self):
        path = self.build_log_path if self.state == "building" else self.log_path
        try:
            with path.open(encoding="utf-8", errors="replace") as file:
                return "".join(deque(file, maxlen=35))[-10000:]
        except OSError:
            return "No logs yet."
