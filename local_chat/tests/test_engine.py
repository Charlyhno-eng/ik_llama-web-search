import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from local_chat.config import AppError, Settings
from local_chat.engine import Engine, process_options


class EngineTests(unittest.TestCase):
    def test_failed_process_load_releases_lock_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "incomplete.gguf"
            model.write_bytes(b"GGUF" + struct.pack("<I", 3))
            engine = Engine(Settings(directory))
            # A process that fails immediately exercises cleanup without a real model.
            with patch.object(engine, "ensure_executable", return_value=sys.executable):
                for _ in range(2):
                    engine.load({"model_path": str(model), "context_size": 4096, "threads": 4})
                    self.assertTrue(engine.operation.acquire(timeout=3))
                    engine.operation.release()
                    self.assertEqual(engine.state, "error")
                    self.assertIsNone(engine.active)
                    self.assertIsNone(engine.process)
            engine.close()

    def test_explicit_missing_binary_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Settings(directory))
            with patch.dict(os.environ, {"LLAMA_SERVER": str(Path(directory) / "missing")}):
                with self.assertRaisesRegex(AppError, "LLAMA_SERVER"):
                    engine.ensure_executable()

    def test_windows_executable_discovery_in_release_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "build/bin/Release/llama-server.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fixture")
            binary.chmod(0o700)
            engine = Engine(Settings(root / "settings"))
            with (
                patch("local_chat.engine.ROOT", root),
                patch("local_chat.engine.sys.platform", "win32"),
                patch.dict(os.environ, {"LLAMA_SERVER": ""}),
            ):
                self.assertEqual(engine.executable(), str(binary))

    def test_platform_process_isolation(self):
        with patch("local_chat.engine.sys.platform", "linux"):
            self.assertEqual(process_options(), {"start_new_session": True})
        with (
            patch("local_chat.engine.sys.platform", "win32"),
            patch.object(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, create=True),
        ):
            self.assertEqual(process_options(), {"creationflags": 512})

    def test_windows_shutdown_targets_owned_tree_and_has_fallback(self):
        for failure in (None, FileNotFoundError("taskkill")):
            with self.subTest(failure=failure):
                process = Mock(pid=12345)
                process.poll.return_value = None
                with (
                    patch("local_chat.engine.sys.platform", "win32"),
                    patch("local_chat.engine.subprocess.run", side_effect=failure) as run,
                ):
                    Engine._stop_process(process)
                self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "12345", "/T", "/F"])
                process.kill.assert_called_once()
                process.wait.assert_called_once_with(timeout=5)

    def test_windows_build_does_not_require_compiler_on_path(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Settings(directory))
            builder = Mock(returncode=0)
            builder.poll.return_value = 0
            with (
                patch("local_chat.engine.sys.platform", "win32"),
                patch.object(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, create=True),
                patch.object(engine, "executable", side_effect=[None, "llama-server.exe"]),
                patch(
                    "local_chat.engine.shutil.which",
                    side_effect=lambda name: "cmake" if name == "cmake" else None,
                ),
                patch("local_chat.engine.subprocess.Popen", return_value=builder) as popen,
            ):
                self.assertEqual(engine.ensure_executable(), "llama-server.exe")
                self.assertEqual(popen.call_count, 2)
                command = popen.call_args.args[0]
                self.assertIn("Release", command)
                self.assertEqual(popen.call_args.kwargs["creationflags"], 512)


if __name__ == "__main__":
    unittest.main()
