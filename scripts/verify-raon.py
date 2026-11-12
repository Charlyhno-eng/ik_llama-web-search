#!/usr/bin/env python3
"""Check a real Raon GGUF through the app loader, chat streaming and retrieval.

python3 scripts/verify-raon.py /path/raon-1-nano-Q4_K_M.gguf report.json --web
Build artifacts use the usual application cache; saved settings are unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_chat.backends import model_architecture, RAON_REVISION
from local_chat.chat import Job, generate
from local_chat.config import AppError, DEFAULT_SETTINGS, Settings
from local_chat.engine import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()
    assert model_architecture(args.model) == "raon1nano"
    report = {"model": str(Path(args.model).resolve()), "revision": RAON_REVISION, "chat": []}
    # Resolve/build the real backend before using disposable settings.
    cache = Engine(Settings())
    try:
        binary = cache.ensure_executable(args.model)
    finally:
        cache.close()
    with tempfile.TemporaryDirectory(prefix="verify-raon-") as temporary:
        engine = Engine(Settings(temporary))
        try:
            with patch.dict("os.environ", {"LLAMA_SERVER": binary}):
                engine.load({**DEFAULT_SETTINGS, "model_path": args.model,
                             "temperature": 0, "max_response_tokens": 128})
                deadline = time.monotonic() + 600
                while engine.state not in ("ready", "error") and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert engine.state == "ready", (engine.message, engine.logs())
            def chat(messages, web=False):
                events = []
                generate(engine, messages, web, Job("verify"),
                         lambda kind, data: events.append((kind, data)))
                text = "".join(data["text"] for kind, data in events if kind == "token")
                done = [data for kind, data in events if kind == "done"]
                assert text and len(done) == 1 and done[0]["metrics"]["engine_timing"], events
                row = {"question": messages[-1]["content"], "answer": text, **done[0]}
                report["chat"].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                return text, events

            messages = []
            for question, expected in [
                ("Réponds uniquement avec la capitale de la France.", "Paris"),
                ("Dans quel pays se trouve cette ville ?", "France"),
            ]:
                messages.append({"role": "user", "content": question})
                answer, _ = chat(messages)
                assert expected.casefold() in answer.casefold(), answer
                messages.append({"role": "assistant", "content": answer})
            job = Job("cancel-check")
            try:
                generate(engine, [{"role": "user", "content": "Compte de 1 à 100."}],
                         False, job, lambda kind, data: job.cancel() if kind == "token" else None)
            except AppError as exc:
                assert exc.status == 499, exc
                report["cancellation"] = "passed"
            else:
                raise AssertionError("Cancellation was ignored")
            answer, _ = chat([{"role": "user", "content": "Combien font 2 + 2 ? Réponds par un chiffre."}])
            assert "4" in answer, answer
            if args.web:
                _, events = chat([{"role": "user", "content":
                    "D’après https://example.com/, quel est le titre de cette page ? Réponds en français."}], True)
                metadata = next(data for kind, data in events if kind == "meta")
                assert metadata["sources"], metadata
                report["web"] = metadata
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        finally:
            engine.close()


if __name__ == "__main__":
    main()
