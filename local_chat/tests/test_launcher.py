"""Exercise the shared launcher without a model or platform shell."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from local_chat.config import ROOT


class LauncherTests(unittest.TestCase):
    def test_launcher_from_another_directory_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="local chat launcher ") as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "start.py"), "--help"],
                cwd=Path(directory),
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Local Chat", result.stdout)
            self.assertIn("--no-browser", result.stdout)
            self.assertIn("--port", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_launcher_forwards_and_validates_arguments(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "start.py"), "--port", "0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("port must be between", result.stderr)


if __name__ == "__main__":
    unittest.main()
