#!/usr/bin/env python3
"""Measure Raon and compare previous/current app loaders on all other local GGUFs.

python3 scripts/benchmark-raon-regression.py MODEL_DIRECTORY OUTPUT_DIRECTORY
Requires a clean baseline engine from --baseline-ref (default HEAD), the existing
bundled binary and a previously built Raon backend. Saved settings are untouched.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_chat.config import DEFAULT_SETTINGS, Settings
from local_chat.engine import Engine
from local_chat.backends import RAON_REVISION

bench = importlib.import_module("benchmark-local")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def loaded(cls, model, directory, shared_cache=None):
    engine = cls(Settings(directory))
    if shared_cache:
        engine.ensure_raon_executable = lambda: shared_cache.ensure_raon_executable()
    try:
        start = time.monotonic()
        engine.load({**DEFAULT_SETTINGS, "model_path": str(model),
                     "temperature": 0, "max_response_tokens": 128})
        deadline = start + 600
        while engine.state not in ("ready", "error") and time.monotonic() < deadline:
            time.sleep(0.1)
        assert engine.state == "ready", (engine.message, engine.logs())
        yield engine, time.monotonic() - start
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-ref", default="HEAD")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "measurements.jsonl"
    results.touch(exist_ok=False)
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:local_chat/engine.py"], cwd=ROOT, text=True)
    (args.output / "engine-before.py").write_text(baseline_source)
    baseline = types.ModuleType("local_chat.engine_before")
    baseline.__package__ = "local_chat"
    exec(compile(baseline_source, str(args.output / "engine-before.py"), "exec"), baseline.__dict__)
    metadata = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        "baseline_loader_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "current_loader_sha256": digest(ROOT / "local_chat/engine.py"),
        "native_source_diff": subprocess.check_output(
            ["git", "diff", args.baseline_ref, "--", "src", "ggml", "common"], cwd=ROOT, text=True),
        "raon_revision": RAON_REVISION,
        "binaries": {str(p.relative_to(ROOT)): digest(p) for p in [
            ROOT / "build/bin/llama-server", ROOT / "build/src/libllama.so",
            ROOT / "build/ggml/src/libggml.so"]},
        "loads": [],
    }
    cache = Engine(Settings())
    all_rows = []
    raon = args.models / "raon-1-nano-Q4_K_M.gguf"
    others = sorted(p for p in args.models.glob("*.gguf") if p != raon)
    try:
        for index, model in enumerate([raon] + others):
            variants = [("raon", Engine)] if model == raon else [
                ("before", baseline.Engine), ("after", Engine)]
            if index % 2:
                variants.reverse()
            for name, cls in variants:
                with loaded(cls, model, args.output / (model.stem + "-" + name), cache) as (engine, load_seconds):
                    command = engine.process.args
                    metadata["loads"].append({"model": model.name, "variant": name,
                                              "load_seconds": load_seconds, "command": command})
                    rows = bench.measure({"lengths": [64, 512, 2048] if model == raon else [64],
                                          "predict": 128, "repeats": 3},
                                         [({"name": name}, engine.port, command)], str(model), results)
                    all_rows.extend(rows)
                    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        cache.close()
    summary = []
    for model in [raon] + others:
        lengths = sorted({r["prompt_length"] for r in all_rows if r["model"] == model.name})
        for length in lengths:
            row = {"model": model.name, "prompt_tokens": length}
            for variant in ("raon", "before", "after"):
                samples = [r for r in all_rows if r["model"] == model.name
                           and r["variant"] == variant and r["prompt_length"] == length and not r["warmup"]]
                if samples:
                    row[variant] = {"decode_median": statistics.median(r["timings"]["predicted_per_second"] for r in samples),
                                    "decode_min": min(r["timings"]["predicted_per_second"] for r in samples),
                                    "decode_max": max(r["timings"]["predicted_per_second"] for r in samples),
                                    "prompt_median": statistics.median(r["timings"]["prompt_per_second"] for r in samples),
                                    "hashes": sorted({r["sha256"] for r in samples})}
            if "before" in row:
                row["change_percent"] = 100 * (row["after"]["decode_median"] / row["before"]["decode_median"] - 1)
                row["same_output"] = row["before"]["hashes"] == row["after"]["hashes"]
                commands = [dict(load) for load in metadata["loads"] if load["model"] == model.name]
                for load in commands:
                    cmd = load["command"]
                    cmd[cmd.index("--port") + 1] = "PORT"
                row["same_command"] = commands[0]["command"] == commands[1]["command"]
            summary.append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
