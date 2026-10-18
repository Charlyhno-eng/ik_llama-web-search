"""HTTP and streaming integration against a deterministic fake inference service."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from local_chat.config import Settings
from local_chat.engine import Engine
from local_chat.server import Application


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            self.respond({"prompt": json.dumps(body["messages"])})
        elif self.path == "/tokenize":
            self.respond({"tokens": list(range(len(body["content"]) // 4))})
        elif not body.get("stream"):
            self.server.last_plan = body
            if "slowplan" in body["messages"][-1]["content"]:
                self.server.prefill_started.set()
                self.server.prefill_release.wait(timeout=5)
            try:
                self.respond(
                    {
                        "choices": [
                            {"message": {"content": "Jev TypeSafe 1.14 release availability"}}
                        ]
                    }
                )
            except OSError:
                pass
        else:
            self.server.last_request = body
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if "prefill" in body["messages"][-1]["content"]:
                self.server.prefill_started.set()
                self.server.prefill_release.wait(timeout=5)
            long_reply = "slow" in body["messages"][-1]["content"]
            try:
                for text in ["Texte "] * 100 if long_reply else ["Comparaison ", "[1]."]:
                    self.wfile.write(
                        (
                            "data: "
                            + json.dumps({"choices": [{"delta": {"content": text}}]})
                            + "\n\n"
                        ).encode()
                    )
                    self.wfile.flush()
                    if long_reply:
                        time.sleep(0.03)
                final = {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "timings": {"predicted_n": 19, "predicted_per_second": 42.5},
                    "usage": {"completion_tokens": 19},
                }
                if "fallback-timing" in body["messages"][-1]["content"]:
                    final["timings"] = {"predicted_n": -1}
                self.wfile.write(("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode())
            except OSError:
                pass

    def respond(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(self.temp.name)
        self.engine = Engine(self.settings)
        self.engine.active = {"model_path": "/model.gguf", "context_size": 4096, "threads": 4}
        self.engine.state = "ready"
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream.prefill_started = threading.Event()
        self.upstream.prefill_release = threading.Event()
        self.engine.port = self.upstream.server_port
        self.app = Application(("127.0.0.1", 0), self.engine)
        self.threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (self.upstream, self.app)
        ]
        for thread in self.threads:
            thread.start()

    def tearDown(self):
        self.app.cancel()
        self.upstream.prefill_release.set()
        for server in (self.app, self.upstream):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, data=None, token=True, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=10)
        request_headers = {"Content-Type": "application/json"}
        if token:
            request_headers["X-Local-Chat-Token"] = self.app.token
        request_headers.update(headers or {})
        connection.request(
            "POST" if data is not None else "GET",
            path,
            body=None if data is None else json.dumps(data),
            headers=request_headers,
        )
        response = connection.getresponse()
        result = (response.status, response.read(), dict(response.getheaders()))
        connection.close()
        return result

    def test_page_bootstrap_static_paths_and_api_protection(self):
        code, page, headers = self.request("/", token=False)
        self.assertEqual(code, 200)
        self.assertIn(self.app.token.encode(), page)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(self.request("/api/status", token=False)[0], 403)
        self.assertEqual(
            self.request("/api/status", headers={"Origin": "https://evil.test"})[0], 403
        )
        self.assertEqual(self.request("/api/status", headers={"Host": "evil.test"})[0], 403)
        self.assertEqual(self.request("/../../LICENSE")[0], 404)
        self.assertEqual(self.request("/api/status")[0], 200)

    def test_stream_sources_budget_and_no_url_generation(self):
        sources = [
            {
                "id": 1,
                "title": "Source réelle",
                "snippet": "Un extrait.",
                "url": "https://example.org/fact",
            }
        ]
        with patch(
            "local_chat.chat.research",
            return_value=(sources, {"queries": ["comparison"], "warnings": []}),
        ):
            code, body, _ = self.request(
                "/api/chat",
                {
                    "id": "source-test",
                    "web": True,
                    "messages": [{"role": "user", "content": "Compare les sources."}],
                },
            )
        stream = body.decode()
        self.assertEqual(code, 200)
        self.assertIn("event: meta", stream)
        self.assertIn("https://example.org/fact", stream)
        self.assertIn("event: token", stream)
        self.assertIn("event: done", stream)
        done = json.loads(stream.split("event: done\ndata: ")[1].split("\n")[0])
        self.assertEqual(done["metrics"]["generated_tokens"], 19)  # Not the two SSE text chunks.
        self.assertEqual(done["metrics"]["tokens_per_second"], 42.5)
        self.assertTrue(done["metrics"]["engine_timing"])
        self.assertGreaterEqual(
            done["metrics"]["first_token_ms"], done["metrics"]["model_first_token_ms"]
        )
        request = self.upstream.last_request
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request["max_tokens"], 1024)
        self.assertNotIn("https://example.org/fact", request["messages"][-1]["content"])
        self.assertTrue(self.engine.operation.acquire(blocking=False))
        self.engine.operation.release()

    def test_generation_uses_saved_temperature_and_response_limit(self):
        self.engine.active.update({"temperature": 0.17, "max_response_tokens": 256})
        code, stream, _ = self.request(
            "/api/chat",
            {"id": "custom-generation", "web": False, "messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(code, 200)
        self.assertIn(b"event: done", stream)
        self.assertEqual(self.upstream.last_request["temperature"], 0.17)
        self.assertEqual(self.upstream.last_request["max_tokens"], 256)

    def test_failed_search_reports_error_without_inference(self):
        from local_chat.config import AppError

        with patch("local_chat.chat.research", side_effect=AppError("DuckDuckGo unavailable", 502)):
            _, stream, _ = self.request(
                "/api/chat",
                {
                    "id": "offline",
                    "web": True,
                    "messages": [{"role": "user", "content": "Question"}],
                },
            )
        self.assertIn(b"event: error", stream)
        self.assertNotIn(b"event: done", stream)
        self.assertFalse(hasattr(self.upstream, "last_request"))

    def test_cancellation_releases_engine_and_rejects_parallel_load(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=5)
        connection.request(
            "POST",
            "/api/chat",
            json.dumps(
                {
                    "id": "cancel-test",
                    "web": False,
                    "messages": [{"role": "user", "content": "slow"}],
                }
            ),
            {"Content-Type": "application/json", "X-Local-Chat-Token": self.app.token},
        )
        response = connection.getresponse()
        while b"event: token" not in response.readline():
            pass
        self.assertFalse(self.engine.operation.acquire(blocking=False))
        self.assertEqual(
            self.request(
                "/api/chat",
                {"id": "busy", "web": False, "messages": [{"role": "user", "content": "hello"}]},
            )[0],
            409,
        )
        self.request("/api/cancel", {"id": "wrong-id"})
        self.assertFalse(self.app.job.cancelled.is_set())
        self.request("/api/cancel", {"id": "cancel-test"})
        self.assertIn(b"event: cancelled", response.read())
        connection.close()
        self.assertTrue(self.engine.operation.acquire(timeout=3))
        self.engine.operation.release()

    def test_invalid_model_does_not_replace_saved_settings(self):
        before = self.settings.value.copy()
        code, _, _ = self.request(
            "/api/settings", {"model_path": "/missing.txt", "context_size": 4096}
        )
        self.assertEqual(code, 400)
        self.assertEqual(self.settings.value, before)

    def test_cancel_during_silent_prefill(self):
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                self.request(
                    "/api/chat",
                    {
                        "id": "silent",
                        "web": False,
                        "messages": [{"role": "user", "content": "prefill"}],
                    },
                )
            )
        )
        thread.start()
        self.assertTrue(self.upstream.prefill_started.wait(timeout=3))
        self.request("/api/cancel", {"id": "silent"})
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "Cancellation must interrupt a silent upstream socket")
        self.assertIn(b"event: cancelled", result[0][1])

    def test_contextual_followup_uses_planner_and_refreshes_previous_source(self):
        url = "https://typesafe.ai/blog/jev-1-14"
        page = {
            "url": url,
            "title": "Jev 1.14 by TypeSafe",
            "blocks": [
                "TypeSafe announces Jev version 1.14, available for use under the preview alias."
            ],
            "links": [],
            "published": "2026-10-14",
        }
        with (
            patch("local_chat.search.search", return_value=[]),
            patch("local_chat.search.read_page", return_value=page),
        ):
            code, stream, _ = self.request(
                "/api/chat",
                {
                    "id": "follow-up",
                    "web": True,
                    "previous_sources": [{"title": page["title"], "url": url}],
                    "messages": [
                        {"role": "user", "content": "Qu'est le modèle Jev de TypeSafe ?"},
                        {"role": "assistant", "content": "Jev est un modèle de décision."},
                        {"role": "user", "content": "y a-t-il une version 1.14 de dispo ?"},
                    ],
                },
            )
        self.assertEqual(code, 200)
        self.assertIn(b"event: done", stream)
        self.assertIn(b"1.14", stream)
        self.assertEqual(self.upstream.last_plan["max_tokens"], 96)
        self.assertIn("français", self.upstream.last_request["messages"][0]["content"])
        self.assertIn("Jev version 1.14", self.upstream.last_request["messages"][-1]["content"])

    def test_cancel_during_query_rewriting(self):
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                self.request(
                    "/api/chat",
                    {
                        "id": "rewrite",
                        "web": True,
                        "messages": [
                            {"role": "user", "content": "What is Jev TypeSafe slowplan?"},
                            {"role": "assistant", "content": "A model."},
                            {"role": "user", "content": "Is version 1.14 available?"},
                        ],
                    },
                )
            )
        )
        thread.start()
        self.assertTrue(self.upstream.prefill_started.wait(timeout=3))
        self.request("/api/cancel", {"id": "rewrite"})
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertIn(b"event: cancelled", result[0][1])
        self.assertFalse(hasattr(self.upstream, "last_request"))

    def test_invalid_prior_source_hints_are_rejected(self):
        for hints in (
            "not a list",
            [None],
            [{"url": "file:///etc/passwd"}],
            [{"url": "https://example.org", "title": 123}],
        ):
            with self.subTest(hints=hints):
                code, _, _ = self.request(
                    "/api/chat",
                    {
                        "id": "bad-hints",
                        "web": False,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "previous_sources": hints,
                    },
                )
                self.assertEqual(code, 400)

    def test_missing_engine_timing_uses_token_usage_not_chunk_count(self):
        _, body, _ = self.request(
            "/api/chat",
            {
                "id": "fallback-timing",
                "web": False,
                "messages": [{"role": "user", "content": "fallback-timing"}],
            },
        )
        done = json.loads(body.decode().split("event: done\ndata: ")[1].split("\n")[0])
        self.assertEqual(done["metrics"]["generated_tokens"], 19)
        self.assertFalse(done["metrics"]["engine_timing"])
        self.assertGreater(done["metrics"]["tokens_per_second"], 0)

    def test_version_without_evidence_does_not_trigger_answer_generation(self):
        older = {
            "url": "https://typesafe.ai/intro",
            "title": "Introducing Jev by TypeSafe",
            "snippet": "Jev version 1.13 is available.",
        }
        old_page = {
            "url": older["url"],
            "title": older["title"],
            "links": [],
            "published": "",
            "blocks": [
                "TypeSafe introduces Jev version 1.13, which is available to all users today."
            ],
        }
        with (
            patch("local_chat.search.search", return_value=[older]),
            patch("local_chat.search.read_page", return_value=old_page),
        ):
            _, stream, _ = self.request(
                "/api/chat",
                {
                    "id": "missing-version",
                    "web": True,
                    "messages": [
                        {"role": "user", "content": "Is Jev TypeSafe version 1.14 available?"}
                    ],
                },
            )
        self.assertIn(b"cannot be confirmed or ruled out", stream)
        self.assertIn(b"event: error", stream)
        self.assertNotIn(b"event: token", stream)
        self.assertFalse(hasattr(self.upstream, "last_request"))


if __name__ == "__main__":
    unittest.main()
