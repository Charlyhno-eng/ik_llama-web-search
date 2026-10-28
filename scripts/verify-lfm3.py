#!/usr/bin/env python3
"""Real-GGUF integration check; no downloads or changes to saved app settings.

python scripts/verify-lfm3.py /path/LFM3-1.7B-Q4_K_M.gguf results.json \
    --reference /path/to/upstream/llama-server
Uses the actual app loader, tokenizer, templates, conversation budgeting and SSE.
Checks reset/repeated requests and prompt chunking around the 256-token SWA
boundary. An optional LFM3-capable upstream build checks next-token probabilities
on the exact same token IDs (absolute tolerance 0.002 for CPU kernel differences).
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_chat.chat import Job, generate
from local_chat.config import AppError, DEFAULT_SETTINGS, Settings
from local_chat.engine import Engine

bench = importlib.import_module("benchmark-local")


def probe(port, tokens, cache=False):
    result = bench.call(port, "/completion", {
        "prompt": tokens, "n_predict": 1, "temperature": 0,
        "cache_prompt": cache, "n_probs": 64, "seed": 42,
    })
    assert result["timings"]["predicted_n"] == 1, result
    entry = result["completion_probabilities"][0]
    if "probs" in entry:
        return {item["tok_str"]: item["prob"] for item in entry["probs"]}
    return {item["token"]: math.exp(item["logprob"]) for item in entry["top_logprobs"]}


def compare(actual, expected, tolerance):
    difference = max(abs(actual.get(k, 0) - expected.get(k, 0))
                     for k in actual.keys() | expected.keys())
    assert difference <= tolerance, (difference, actual, expected)
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": str(Path(args.model).resolve()), "probes": [], "chat": []}
    with tempfile.TemporaryDirectory(prefix="verify-lfm3-") as temporary:
        engine = Engine(Settings(temporary))
        try:
            engine.load({**DEFAULT_SETTINGS, "model_path": args.model,
                         "temperature": 0, "max_response_tokens": 128})
            deadline = time.monotonic() + 600
            while engine.state not in ("ready", "error") and time.monotonic() < deadline:
                time.sleep(0.1)
            assert engine.state == "ready", (engine.message, engine.logs())
            job = Job("cancel-check")
            def cancel_first_token(kind, data):
                if kind == "token":
                    job.cancel()
            try:
                generate(engine, [{"role": "user", "content": "Compte de 1 à 100."}],
                         False, job, cancel_first_token)
            except AppError as exc:
                assert exc.status == 499, exc
                report["cancellation"] = "stopped after first text; subsequent chat checks recovery"
            else:
                raise AssertionError("Cancellation was ignored")
            messages = []
            for question, answer in [("Réponds uniquement avec la capitale de la France.", "Paris"),
                                     ("Dans quel pays se trouve cette ville ?", "France")]:
                messages.append({"role": "user", "content": question})
                events = []
                generate(engine, messages, False, Job("verify"),
                         lambda kind, data: events.append((kind, data)))
                text = "".join(data["text"] for kind, data in events if kind == "token")
                assert answer.lower() in text.lower(), text
                done = [data for kind, data in events if kind == "done"]
                assert len(done) == 1 and done[0]["metrics"]["engine_timing"], events
                messages.append({"role": "assistant", "content": text})
                report["chat"].append({"question": question, "answer": text, **done[0]})
            tokens = engine.call("/tokenize", {"content": bench.QUESTION + bench.PADDING * 150})["tokens"]
            cases = {n: tokens[:n] for n in (64, 255, 256, 257, 512, 1024)}
            results = {}
            for n, prompt in cases.items():
                results[n] = probe(engine.port, prompt)
                # Repeat after a different sequence to expose stale convolution state.
                probe(engine.port, tokens[10:40])
                error = compare(probe(engine.port, prompt), results[n], 0.002)
                report["probes"].append({"tokens": n, "reset_max_error": error})
            # Chunked prefill must represent the same recurrent/SWA state.
            variant = {"name": "chunked", "args": ["-ub", "32"]}
            with bench.server(variant, args.model, args.output, 0) as (port, _):
                for row in report["probes"]:
                    n = row["tokens"]
                    row["chunk_max_error"] = compare(probe(port, cases[n]), results[n], 0.002)
            if args.reference:
                variant = {"name": "upstream", "runtime": "llama.cpp", "binary": args.reference}
                with bench.server(variant, args.model, args.output, 1) as (port, _):
                    for row in report["probes"]:
                        n = row["tokens"]
                        row["reference_max_error"] = compare(probe(port, cases[n]), results[n], 0.002)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        finally:
            engine.close()


if __name__ == "__main__":
    main()
