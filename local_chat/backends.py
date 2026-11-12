"""Pinned inference backend for architectures newer than the bundled engine."""

import struct

from .config import AppError

# ik_llama.cpp PR #2903; keep the revision fixed so builds are reproducible.
RAON_REPOSITORY = "https://github.com/Bl4ckyou/ik_llama.cpp.git"
RAON_REVISION = "b87a4519321a3430cba206fd130a367ef7f9fb5d"


def model_architecture(path):
    """Read GGUF metadata without loading tensors or requiring numpy."""
    with open(path, "rb") as file:
        def read(size):
            data = file.read(size)
            if len(data) != size:
                raise ValueError("truncated metadata")
            return data

        def integer(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

        def string():
            size = integer("Q")
            if size > 64 * 1024 * 1024:
                raise ValueError("oversized metadata string")
            return read(size).decode("utf-8")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(kind, depth=0):
            if kind in sizes:
                read(sizes[kind])
            elif kind == 8:
                string()
            elif kind == 9 and depth < 2:
                subtype, count = integer("I"), integer("Q")
                if count > 10_000_000:
                    raise ValueError("oversized metadata array")
                if subtype in sizes:
                    file.seek(sizes[subtype] * count, 1)
                else:
                    for _ in range(count):
                        skip(subtype, depth + 1)
            else:
                raise ValueError("unsupported metadata type")

        try:
            if read(4) != b"GGUF" or integer("I") not in (2, 3):
                raise ValueError("invalid header")
            integer("Q")  # tensor count
            count = integer("Q")
            if count > 100_000:
                raise ValueError("oversized metadata")
            for _ in range(count):
                key, kind = string(), integer("I")
                if key == "general.architecture":
                    if kind != 8:
                        raise ValueError("architecture must be a string")
                    return string()
                skip(kind)
            return ""
        except (ValueError, UnicodeError, struct.error) as exc:
            raise AppError("Unable to read the model's GGUF architecture metadata.") from exc
