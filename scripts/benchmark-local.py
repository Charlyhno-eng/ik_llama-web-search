#!/usr/bin/env python3
"""Compare owned local servers with identical, uncached token sequences.

Usage: python scripts/benchmark-local.py config.json results.jsonl
Configuration example (repeat variants in reverse order for an ABBA comparison):
{
  "models": ["/path/to/model.gguf"],
  "lengths": [64, 512, 2048], "predict": 128, "repeats": 3,
  "variants": [
    {"name": "before", "binary": "/path/to/baseline/llama-server",
     "lib": "/path/to/baseline/libraries", "threads": 4},
    {"name": "after", "binary": "build/bin/llama-server", "threads": 4}
  ]
}
Set "interleave": true with two variants to keep both servers loaded and alternate
requests in ABBA order (one generation at a time). Otherwise servers run serially.
Variants also accept threads_batch, args (additional server arguments), and env.
Use runtime="llama.cpp" for an upstream server (omits ik-specific flags).
Optional prompt_text and padding_text replace the default synthetic French prompt.
Alternatively messages applies the model's chat template and uses the complete prompt.
chat_template_kwargs passes template options such as enable_thinking to that template.
Only the standard library and an existing model/server are required. Each server
is stopped in a finally block. Warmups are recorded but excluded from summaries.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener


HTTP = build_opener(ProxyHandler({}))
QUESTION = (
    "Tu es un assistant utile. Explique en français comment un ordinateur exécute "
    "un programme, avec des exemples et des étapes détaillées. Voici le contexte : "
)
PADDING = "Le processeur lit les instructions et utilise la mémoire pour effectuer les calculs. "


def call(port, path, data=None):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with HTTP.open(request, timeout=300) as response:
        return json.load(response)


@contextmanager
def server(variant, model, output, index):
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    env = os.environ.copy()
    if variant.get("lib"):
        env["LD_LIBRARY_PATH"] = str(Path(variant["lib"]).resolve())
    env.update(variant.get("env", {}))
    threads = variant.get("threads", 4)
    binary = str(Path(variant.get("binary", "build/bin/llama-server")).resolve())
    command = [
        binary, "-m", model, "-c", "4096", "-t", str(threads),
        "-tb", str(variant.get("threads_batch", threads)), "--jinja",
        "--host", "127.0.0.1", "--port", str(port),
        "-np", "1", "--no-context-shift",
    ]
    if variant.get("runtime", "ik_llama") == "ik_llama":
        command += ["-rtr", "--reasoning-tokens", "none"]
    command += variant.get("args", [])
    label = re.sub(r"[^a-zA-Z0-9_.-]", "_", variant["name"] + "-" + Path(model).stem)
    log_path = output.parent / f"{output.stem}-{index}-{label}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited ({process.returncode}); see {log_path}")
                try:
                    call(port, "/health")
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Server did not become ready; see {log_path}")
                    time.sleep(0.2)
            yield port, command
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def measure(config, servers, model, output):
    rows = []
    port = servers[0][1]
    text = config.get("prompt_text", QUESTION)
    if "messages" in config:
        text = call(port, "/apply-template", {
            "messages": config["messages"],
            "chat_template_kwargs": config.get("chat_template_kwargs", {}),
        })["prompt"]
    base = call(port, "/tokenize", {"content": text})["tokens"]
    padding = call(port, "/tokenize", {"content": config.get("padding_text", PADDING) * 700})["tokens"]
    lengths = [len(base)] if "messages" in config else config.get("lengths", [64, 512, 2048])
    for length in lengths:
        if length + config.get("predict", 128) > 4096:
            raise ValueError("Prompt and generated tokens exceed the benchmark context")
        prompt = (base + padding)[:length]
        if len(prompt) != length:
            raise ValueError("Not enough padding tokens for requested prompt length")
        for run in range(config.get("repeats", 3) + 1):
            order = servers if run % 2 == 0 else servers[::-1]
            for variant, port, command in order:
                start = time.monotonic()
                response = call(port, "/completion", {
                    "prompt": prompt, "n_predict": config.get("predict", 128),
                    "temperature": config.get("temperature", 0), "seed": 42, "ignore_eos": True,
                    "cache_prompt": False,
                })
                elapsed = time.monotonic() - start
                timings = response["timings"]
                if timings["prompt_n"] != length:
                    raise RuntimeError("Prompt was reused or its length changed")
                if timings["predicted_n"] != config.get("predict", 128):
                    raise RuntimeError("Server did not generate the requested token count")
                speed = timings["predicted_per_second"]
                if not math.isfinite(speed) or speed <= 0:
                    raise RuntimeError("Invalid engine timing")
                row = {
                    "variant": variant["name"], "model": Path(model).name,
                    "case": config.get("case", "synthetic"),
                    "prompt_length": length, "run": run, "warmup": run == 0,
                    "elapsed": elapsed, "timings": timings,
                    "tokens_predicted": response.get("tokens_predicted"),
                    "temperature": config.get("temperature", 0), "seed": 42,
                    "sha256": hashlib.sha256(response["content"].encode()).hexdigest(),
                    "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                    "command": command, "env_overrides": variant.get("env", {}),
                    "library_path": variant.get("lib"),
                }
                with output.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
                rows.append(row)
                print(f"{variant['name']} {Path(model).name} prompt={length} "
                      f"run={run}: {speed:.3f} tok/s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, help="New JSONL file; existing results are never overwritten")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lengths = config.get("lengths", [64, 512, 2048])
    predict = config.get("predict", 128)
    repeats = config.get("repeats", 3)
    if (not config.get("models") or not config.get("variants") or not lengths
            or not isinstance(predict, int) or predict < 1
            or not isinstance(repeats, int) or repeats < 1
            or any(not isinstance(n, int) or n < 1 or n + predict > 4096 for n in lengths)):
        parser.error("Need models, variants, positive repeats/predict, and prompts fitting 4096 tokens")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8"):
        pass
    groups = defaultdict(list)
    indexed = list(enumerate(config["variants"]))
    batches = [indexed] if config.get("interleave", False) else [[v] for v in indexed]
    for model in config["models"]:
        for batch in batches:
            with ExitStack() as stack:
                servers = [
                    (variant, *stack.enter_context(server(variant, model, args.output, index)))
                    for index, variant in batch
                ]
                for row in measure(config, servers, model, args.output):
                    if not row["warmup"]:
                        groups[(row["variant"], row["model"], row["prompt_length"])].append(row)
    print("\nMedians (warmups excluded):")
    for key, rows in groups.items():
        speed = statistics.median(row["timings"]["predicted_per_second"] for row in rows)
        prompt_speed = statistics.median(row["timings"]["prompt_per_second"] for row in rows)
        print(f"{key}: decode={speed:.3f} tok/s, prompt={prompt_speed:.3f} tok/s, n={len(rows)}")


if __name__ == "__main__":
    main()
