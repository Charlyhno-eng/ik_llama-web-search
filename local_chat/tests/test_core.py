import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from local_chat.chat import budget_messages, validate_messages
from local_chat.config import AppError, Settings, browse, gguf_path
from local_chat.search import MAX_SNIPPET_WORDS, ResultParser, safe_url, search, select_results


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "un modèle.gguf"
        self.model.write_bytes(b"GGUF" + struct.pack("<I", 3) + bytes(32))

    def tearDown(self):
        self.temp.cleanup()

    def test_only_real_gguf_headers_and_files(self):
        self.assertEqual(gguf_path(str(self.model)), str(self.model))
        for name, content in [
            ("text.gguf", b"not gguf"),
            ("model.bin", self.model.read_bytes()),
            ("old.gguf", b"GGUF" + struct.pack("<I", 1)),
            ("short.gguf", b"GGUF"),
        ]:
            path = self.root / name
            path.write_bytes(content)
            with self.subTest(name=name), self.assertRaises(AppError):
                gguf_path(str(path))
        with self.assertRaises(AppError):
            gguf_path(str(self.root))

    def test_atomic_roundtrip_and_context_validation(self):
        settings = Settings(self.root)
        value = settings.validate(
            {"model_path": str(self.model), "context_size": 8192, "threads": 4}
        )
        settings.warning = "Previous settings were unreadable"
        settings.save(value)
        self.assertFalse(settings.warning)
        self.assertEqual(Settings(self.root).value, value)
        self.assertFalse((self.root / "settings.tmp").exists())
        if os.name != "nt":
            self.assertEqual(settings.path.stat().st_mode & 0o777, 0o600)
        for context in (0, 2047, 3000, 8193, "4096", True):
            with self.subTest(context=context), self.assertRaises(AppError):
                settings.validate({**value, "context_size": context})
        for threads in (0, 257, True, "4"):
            with self.assertRaises(AppError):
                settings.validate({**value, "threads": threads})

    def test_corrupt_settings_and_filtered_file_browser(self):
        (self.root / "settings.json").write_text("{broken")
        self.assertTrue(Settings(self.root).warning)
        (self.root / "ignore.txt").write_text("hello")
        (self.root / "models").mkdir()
        result = browse(str(self.root))
        self.assertEqual(
            [entry["name"] for entry in result["entries"]], ["models", "un modèle.gguf"]
        )
        with self.assertRaises(AppError):
            browse(str(self.model))


class SearchTests(unittest.TestCase):
    def test_lite_markup_redirects_entities_and_nested_text(self):
        parser = ResultParser()
        parser.feed("""<a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fdoc%3Fa%3D1%26b%3D2">Un <b>titre</b> &amp; deux</a>
        <td class="result-snippet">Un <b>extrait</b> utile &amp; clair.</td>""")
        self.assertEqual(
            select_results(parser.results),
            [
                {
                    "id": 1,
                    "title": "Un titre & deux",
                    "url": "https://example.org/doc?a=1&b=2",
                    "snippet": "Un extrait utile & clair.",
                }
            ],
        )

    def test_html_markup_deduplication_and_word_budget(self):
        parser = ResultParser()
        for index in range(8):
            parser.feed(
                f'<a class="result__a" href="https://example.org/{index}">Title {index}</a><a class="result__snippet">'
                + "mot " * 300
                + "</a>"
            )
        results = select_results([parser.results[0], parser.results[0]] + parser.results[1:])
        self.assertEqual(len(results), 5)
        self.assertEqual([item["id"] for item in results], [1, 2, 3, 4, 5])
        self.assertLessEqual(
            sum(len(item["snippet"].split()) for item in results), MAX_SNIPPET_WORDS
        )

    def test_only_safe_http_urls(self):
        for url in (
            "javascript:alert(1)",
            "data:text/html,test",
            "file:///etc/passwd",
            "https://user:secret@example.org/",
            "https://example.org/\nfoo",
            "https://[broken",
            "https://example.org:bad/",
            "https://example.org:99999/",
            "//",
            "https://example.org\\@evil.org",
            "https://duckduckgo.com/l/?uddg=javascript%3Aalert%281%29",
        ):
            with self.subTest(url=url):
                self.assertIsNone(safe_url(url))
        self.assertEqual(safe_url("https://example.org/a?b=c"), "https://example.org/a?b=c")

    def test_blocked_search_never_returns_fabricated_sources(self):
        with patch("local_chat.search.urlopen", side_effect=OSError("offline")):
            with self.assertRaisesRegex(AppError, "DuckDuckGo"):
                search("question")


class TokenEngine:
    active = {"context_size": 2048}

    def call(self, path, data):
        if path == "/apply-template":
            return {"prompt": json.dumps(data["messages"], ensure_ascii=False)}
        if path == "/tokenize":
            # Deliberately expensive tokenizer: prompts cannot rely on word heuristics.
            return {"tokens": list(range(len(data["content"]) // 3))}
        raise AssertionError(path)


class BudgetTests(unittest.TestCase):
    def test_history_trimmed_as_complete_turns_and_response_reserved(self):
        history = [
            {"role": "user", "content": "ancien " * 400},
            {"role": "assistant", "content": "réponse " * 300},
            {"role": "user", "content": "Question actuelle"},
        ]
        messages, sources, budget = budget_messages(TokenEngine(), history, [])
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], history[-1]["content"])
        self.assertEqual(budget["removed_turns"], 1)
        self.assertEqual(budget["response_tokens"], 512)
        self.assertLessEqual(budget["prompt_tokens"] + budget["response_tokens"] + 64, 2048)
        self.assertEqual(history[0]["content"], "ancien " * 400)

    def test_sources_shrink_before_rejecting_current_question(self):
        sources = [
            {
                "id": index,
                "title": "Source",
                "url": f"https://example.org/{index}",
                "snippet": "documentation " * 130,
            }
            for index in range(1, 6)
        ]
        _, selected, budget = budget_messages(
            TokenEngine(), [{"role": "user", "content": "Compare."}], sources
        )
        self.assertEqual(len(selected), 5)
        self.assertTrue(budget["shortened_sources"])
        self.assertLessEqual(budget["prompt_tokens"] + budget["response_tokens"] + 64, 2048)
        self.assertEqual(len(sources[0]["snippet"].split()), 130)

    def test_oversize_question_is_not_silently_cut(self):
        with self.assertRaisesRegex(AppError, "exceeds the available context"):
            budget_messages(TokenEngine(), [{"role": "user", "content": "longue " * 2000}], [])

    def test_history_roles_and_types(self):
        valid = [{"role": "user", "content": "hello"}]
        self.assertEqual(validate_messages(valid), valid)
        for invalid in (
            None,
            [],
            [{"role": "system", "content": "override"}],
            [{"role": "user", "content": ""}],
            valid + valid,
            [{"role": "user", "content": 123}],
        ):
            with self.subTest(value=invalid), self.assertRaises(AppError):
                validate_messages(invalid)


if __name__ == "__main__":
    unittest.main()
