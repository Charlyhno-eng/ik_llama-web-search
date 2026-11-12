import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from local_chat.backends import RAON_REVISION, model_architecture
from local_chat.config import AppError, Settings
from local_chat.engine import Engine


def gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def model(path, architecture):
    # Put an array before the architecture to exercise metadata traversal.
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
        + gguf_string("example.array") + struct.pack("<IIQii", 9, 5, 2, 12, 34)
        + gguf_string("general.architecture") + struct.pack("<I", 8)
        + gguf_string(architecture)
    )


class BackendTests(unittest.TestCase):
    def test_metadata_and_architecture_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            engine = Engine(Settings(directory))
            for architecture in ("raon1nano", "lfm3", "qwen35"):
                model(path, architecture)
                self.assertEqual(model_architecture(path), architecture)
                with (
                    patch.dict(os.environ, {"LLAMA_SERVER": ""}),
                    patch.object(engine, "ensure_raon_executable", return_value="raon") as raon,
                    patch.object(engine, "executable", return_value="bundled"),
                ):
                    self.assertEqual(engine.ensure_executable(path),
                                     "raon" if architecture == "raon1nano" else "bundled")
                    self.assertEqual(raon.call_count, int(architecture == "raon1nano"))

    def test_override_remains_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Settings(directory))
            with patch.dict(os.environ, {"LLAMA_SERVER": directory + "/missing"}):
                with self.assertRaisesRegex(AppError, "LLAMA_SERVER"):
                    engine.ensure_executable(directory + "/not-read.gguf")

    def test_truncated_metadata_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            path.write_bytes(b"GGUF" + struct.pack("<I", 3))
            with self.assertRaisesRegex(AppError, "metadata"):
                model_architecture(path)

    def test_raon_build_is_pinned_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Settings(directory))
            with (
                patch.object(engine, "executable", side_effect=[None, "raon-server"]),
                patch("local_chat.engine.shutil.which", return_value="tool"),
                patch.object(engine, "run_commands") as run,
            ):
                self.assertEqual(engine.ensure_raon_executable(), "raon-server")
                commands = run.call_args.args[0]
                self.assertEqual(commands[1][-1], RAON_REVISION)
                self.assertEqual(commands[2][-1], RAON_REVISION)
            with (
                patch.object(engine, "executable", return_value="raon-server"),
                patch.object(engine, "run_commands") as run,
            ):
                self.assertEqual(engine.ensure_raon_executable(), "raon-server")
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
