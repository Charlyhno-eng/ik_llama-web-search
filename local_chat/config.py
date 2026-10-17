"""Small, atomic settings store and validation of local GGUF paths."""

import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent.parent
CONTEXT_SIZES = (2048, 4096, 6144, 8192)


class AppError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def gguf_path(value):
    if not isinstance(value, str) or not value.strip():
        raise AppError("Choose a GGUF model.")
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise AppError("The model must be an existing .gguf file.")
    try:
        with path.open("rb") as file:
            header = file.read(8)
    except OSError as exc:
        raise AppError("Unable to read this GGUF file.") from exc
    if (
        len(header) != 8
        or header[:4] != b"GGUF"
        or struct.unpack("<I", header[4:])[0] not in (2, 3)
    ):
        raise AppError("This file does not have a valid GGUF header (version 2 or 3).")
    return str(path)


class Settings:
    def __init__(self, directory=None):
        self.directory = Path(
            directory or os.environ.get("LOCAL_CHAT_DATA_DIR", ROOT / ".local-chat")
        )
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "settings.json"
        self.value = {"model_path": "", "context_size": 4096, "threads": 4}
        self.warning = ""
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("settings must be an object")
            if saved.get("context_size") in CONTEXT_SIZES:
                self.value["context_size"] = saved["context_size"]
            if type(saved.get("threads")) is int and 1 <= saved["threads"] <= 256:
                self.value["threads"] = saved["threads"]
            if isinstance(saved.get("model_path"), str):
                self.value["model_path"] = saved["model_path"]
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            self.warning = "Saved settings are unreadable. Default values are being used."

    def validate(self, data):
        if type(data.get("context_size")) is not int or data["context_size"] not in CONTEXT_SIZES:
            raise AppError("Context must be 2,048, 4,096, 6,144 or 8,192 tokens.")
        threads = data.get("threads", 4)
        if type(threads) is not int or not 1 <= threads <= 256:
            raise AppError("Thread count must be between 1 and 256.")
        return {
            "model_path": gguf_path(data.get("model_path")),
            "context_size": data["context_size"],
            "threads": threads,
        }

    def save(self, value):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            os.chmod(temporary, 0o600)
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(self.path)
        self.value = value.copy()
        self.warning = ""


def browse(directory=None):
    path = Path(directory or Path.home()).expanduser().resolve()
    if not path.is_dir():
        raise AppError("This folder does not exist.")
    entries = []
    try:
        for item in path.iterdir():
            try:
                if item.name.startswith("."):
                    continue
                is_directory = item.is_dir()
                if is_directory or (item.is_file() and item.suffix.lower() == ".gguf"):
                    entries.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "directory": is_directory,
                            "size": None if is_directory else item.stat().st_size,
                        }
                    )
            except OSError:
                continue
    except OSError as exc:
        raise AppError("This folder cannot be accessed.") from exc
    entries.sort(key=lambda item: (not item["directory"], item["name"].casefold()))
    return {
        "directory": str(path),
        "parent": str(path.parent),
        "entries": entries,
        "home": str(Path.home()),
    }
