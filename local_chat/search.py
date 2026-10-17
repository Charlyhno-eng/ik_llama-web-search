"""Contextual DuckDuckGo search, relevance ranking, and public-page evidence."""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from html.parser import HTMLParser
from http.client import HTTPException
import re
import time
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .config import AppError
from .retrieval import evidence_excerpt, read_page, terms, word_terms

MAX_SNIPPET_WORDS = 750
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
_cache = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SECONDS = 180
FOLLOWUP_WORDS = set(
    """release releases pricing price prices prix cost cout costs limitations limits
limites performances performance features fonctionnalites nouveautes update updates derniere dernier
recent recentement nouvelles nouveaux differences difference comparaison compare comparer explain
explique expliquer tell look found trouve voici pourtant its about""".split()
)


def subject_terms(text):
    words = word_terms(text)
    return words - FOLLOWUP_WORDS or words


def safe_url(value):
    if not isinstance(value, str) or len(value) > 4096 or re.search(r"[\s\x00-\x1f\\]", value):
        return None
    try:
        parts = urlsplit(value if not value.startswith("//") else "https:" + value)
        if parts.hostname in ("duckduckgo.com", "www.duckduckgo.com") and parts.path.startswith(
            "/l/"
        ):
            value = parse_qs(parts.query).get("uddg", [""])[0]
            parts = urlsplit(value)
        if (
            parts.scheme not in ("https", "http")
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            return None
        # Accessing .port validates malformed/non-numeric/out-of-range port values.
        parts.port
        if re.search(r"[\s\x00-\x1f\\]", value):
            return None
        return urlunsplit(parts)
    except ValueError:
        return None


class ResultParser(HTMLParser):
    """Extract both DDG result layouts without depending on third-party markup code."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.capture = None
        self.end_tag = None
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            url = safe_url(attrs.get("href", ""))
            if url:
                self.results.append({"title": "", "url": url, "snippet": ""})
                self.capture, self.end_tag, self.parts = "title", "a", []
        elif self.results and ("result__snippet" in classes or "result-snippet" in classes):
            self.capture, self.end_tag, self.parts = "snippet", tag, []

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if self.capture and tag == self.end_tag:
            self.results[-1][self.capture] = " ".join("".join(self.parts).split())
            self.capture = None


def select_results(results, limit=5):
    selected, seen = [], set()
    for result in results:
        url = safe_url(result.get("url"))
        if not url or not result.get("title") or not result.get("snippet"):
            continue
        parts = urlsplit(url)
        key = (parts.netloc.lower(), parts.path.rstrip("/"), parts.query)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "id": len(selected) + 1,
                "title": result["title"][:240],
                "url": url,
                "snippet": " ".join(result["snippet"].split()[: MAX_SNIPPET_WORDS // limit]),
            }
        )
        if len(selected) == limit:
            break
    return selected


def search(query, limit=5):
    query = " ".join(query.split())[:500]
    if not query:
        raise AppError("The search query is empty.")
    with _cache_lock:
        cached = _cache.get(query)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            _cache.move_to_end(query)
            return select_results(cached[1], limit)
    # Lite is first because the full HTML endpoint more often presents a CAPTCHA.
    for base in ("https://lite.duckduckgo.com/lite/", "https://html.duckduckgo.com/html/"):
        request = Request(
            base + "?" + urlencode({"q": query, "kl": "wt-wt"}),
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en,fr;q=0.8"},
        )
        try:
            with urlopen(request, timeout=6) as response:
                body = response.read(2_000_001)
                if len(body) > 2_000_000:
                    continue
                html = body.decode("utf-8", errors="replace")
            if "anomaly.js" in html or "anomaly-modal" in html:
                continue
            parser = ResultParser()
            parser.feed(html)
            results = select_results(parser.results, limit=limit)
            if results:
                with _cache_lock:
                    _cache[query] = (
                        time.monotonic(),
                        [item.copy() for item in parser.results[:20]],
                    )
                    _cache.move_to_end(query)
                    while len(_cache) > 64:
                        _cache.popitem(last=False)
                return results
        except (HTTPError, URLError, OSError, ValueError):
            continue
    raise AppError(
        "DuckDuckGo returned no usable results (anti-bot protection, network error, or no matches). Retry or turn off web search.",
        502,
    )


def urls_in(text):
    return list(
        dict.fromkeys(
            url
            for match in re.findall(r"https?://[^\s<>\[\]\"']+", text)
            if (url := safe_url(match.rstrip(".,;:!?)}")))
        )
    )[:3]


def compact_query(text):
    text = re.sub(r"https?://\S+", " ", text)
    return " ".join(dict.fromkeys(terms(text)))[:400]


def search_context(messages, previous_sources=()):
    """Deterministic fallback uses user questions, never the assistant's assertions."""
    current = messages[-1]["content"]
    current_terms = word_terms(re.sub(r"https?://\S+", "", current))
    previous_questions = [item["content"] for item in messages[:-1] if item["role"] == "user"]
    # A brief follow-up without a new named subject needs the previous user topic.
    named = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", current)
    named = [name for name in named if word_terms(name)]
    followup = bool(previous_questions) and (
        len(current_terms) < 4
        and not named
        or bool(
            re.search(
                r"\b(et (?:sa|ses|son|pour|la|le)|and (?:its|what|the)|it|that|this|elle|celui|celle|cette|ce modèle|sa version|ses limites)\b",
                current,
                re.I,
            )
        )
    )
    topic = ""
    if followup:
        for question in reversed(previous_questions[-8:]):
            if word_terms(question) - FOLLOWUP_WORDS:
                topic = question
                break
        if not topic and previous_questions:
            topic = previous_questions[-1]
    anchors = subject_terms(topic if topic else re.sub(r"https?://\S+", "", current))
    if followup:
        anchors |= set().union(*(word_terms(name) for name in named)) if named else set()
    # New versions/numbers must come from the current question, not an earlier version.
    topic_query = " ".join(word for word in terms(topic) if not re.search(r"\d", word))
    query = compact_query((topic_query + " " + current).strip()) or current[:400]
    if re.search(r"\b\d+\.\d+\b", current) and re.search(
        r"version|dispo|available|release", current, re.I
    ):
        query += " release"
    direct = urls_in(current)
    if direct:
        anchors = set().union(
            *(
                word_terms(
                    urlsplit(url).hostname.replace(".", " ")
                    + " "
                    + urlsplit(url).path.replace("/", " ").replace("-", " ")
                )
                for url in direct
            )
        ) - {"com", "org", "www", "blog", "docs", "html", "https"}
        parts = urlsplit(direct[0])
        path = re.sub(r"(?<=\d)-(?=\d)", ".", parts.path)
        query = compact_query(
            parts.hostname.replace(".", " ")
            + " "
            + path.replace("/", " ").replace("-", " ")
            + " "
            + current
        )
    hints = []
    for source in previous_sources[-5:]:
        if isinstance(source, dict) and (url := safe_url(source.get("url"))):
            host = urlsplit(url).hostname.removeprefix("www.")
            # Prior domains are query hints only, never fresh evidence or asserted official sites.
            if anchors & word_terms(host.replace(".", " ")):
                hints.append(host)
    return {
        "query": query,
        "anchors": anchors,
        "followup": followup,
        "direct": direct,
        "domains": list(dict.fromkeys(hints))[:1],
        "question": current,
    }


def relevance(result, anchors, query):
    title = word_terms(result.get("title", ""))
    text = set(terms(result.get("snippet", "")))
    url = word_terms(result.get("url", "").replace("/", " ").replace(".", " "))
    hits = anchors & (title | text | url)
    if anchors and not hits:
        return -1
    numbers = {word for word in terms(query) if re.search(r"\d", word)}
    all_text = " ".join(
        (result.get("title", ""), result.get("snippet", ""), result.get("url", ""))
    ).casefold()
    return (
        4 * len(anchors & title)
        + 2 * len(anchors & text)
        + 3 * len(anchors & url)
        + 5 * sum(number in all_text for number in numbers)
    )


def mentions_version(text, version):
    pattern = re.escape(version).replace(r"\.", r"[.\-_]")
    return bool(re.search(r"(?<!\d)" + pattern + r"(?!\d)", text, re.I))


def research(messages, previous_sources=(), rewrite=None, job=None, emit=None):
    """Retrieve afresh on every turn; never fill five slots with unrelated results."""
    plan = search_context(messages, previous_sources)
    if rewrite and plan["followup"] and not plan["direct"]:
        rewritten = rewrite(plan)
        # Rewrites cannot discard the subject or the version requested by the user.
        needed_numbers = set(re.findall(r"\b\d+(?:\.\d+)+\b", plan["question"]))
        if (
            isinstance(rewritten, str)
            and 3 <= len(rewritten) <= 400
            and (
                not plan["anchors"]
                or len(plan["anchors"] & word_terms(rewritten)) >= min(2, len(plan["anchors"]))
            )
            and all(number in rewritten for number in needed_numbers)
            and not urls_in(rewritten)
        ):
            plan["query"] = rewritten.strip()
    deadline = time.monotonic() + 40
    queries = []
    warnings = []
    candidates = []
    pages = {}
    attempted = set()
    failures = []

    def check():
        if job:
            job.check()

    def stage(message):
        check()
        if emit:
            emit("stage", {"message": message})

    def lookup(query):
        check()
        try:
            return search(query, limit=8)
        except AppError as exc:
            if exc.status == 499:
                raise
            failures.append(query)
            return []

    def fetch(result):
        check()
        try:
            page = read_page(result["url"], job=job, deadline=deadline)
            excerpt = evidence_excerpt(page["blocks"], plan["query"], 150)
            if len(excerpt.split()) < 8:
                return None
            return {
                **result,
                "url": page["url"],
                "title": page["title"] or result["title"],
                "snippet": excerpt,
                "kind": "page",
                "published": page["published"],
                "_links": page["links"],
            }
        except (OSError, ValueError, HTTPException):
            return None

    def read_candidates(items):
        check()
        fresh = []
        for item in items:
            if item["url"] not in attempted:
                attempted.add(item["url"])
                fresh.append(item)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for original, fetched in zip(fresh, pool.map(fetch, fresh)):
                if fetched:
                    pages[original["url"]] = fetched
        check()

    # Supplied URLs bypass search indexing: a fresh release may not be indexed yet.
    if plan["direct"]:
        stage("Reading the provided pages…")
        direct = [
            {"url": url, "title": urlsplit(url).hostname, "snippet": "", "_direct": True}
            for url in plan["direct"]
        ]
        read_candidates(direct)
        candidates.extend(page for page in pages.values())
        if any(url not in pages for url in plan["direct"]):
            warnings.append(
                "A provided page could not be read; available search excerpts are used instead."
            )

    # Earlier retrieved URLs are navigation hints, not reusable facts. Read them again now.
    if plan["followup"] and previous_sources:
        seeds = [
            {"url": url, "title": source.get("title", ""), "snippet": ""}
            for source in previous_sources
            if isinstance(source, dict) and (url := safe_url(source.get("url")))
        ]
        seeds = [item for item in seeds if relevance(item, plan["anchors"], plan["query"]) >= 0]
        seeds.sort(key=lambda item: -relevance(item, plan["anchors"], plan["query"]))
        stage("Revisiting relevant sources from this conversation…")
        read_candidates(seeds[:3])
        candidates.extend(pages[item["url"]] for item in seeds[:3] if item["url"] in pages)

    stage("Searching DuckDuckGo with conversation context…")
    initial = [plan["query"]]
    initial += [f"site:{domain} {plan['query']}" for domain in plan["domains"]]
    # A URL's hostname/path is a much better fallback than the surrounding 'this site says'.
    if plan["direct"]:
        parts = urlsplit(plan["direct"][0])
        initial = [f"site:{parts.hostname} {parts.path.replace('/', ' ').replace('-', ' ')}"]
    queries.extend(dict.fromkeys(initial))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for results in pool.map(lookup, queries):
            candidates.extend(results)
    check()

    def ranked():
        unique = {}
        for candidate in candidates:
            item = pages.get(candidate["url"], candidate)
            score = relevance(item, plan["anchors"], plan["query"])
            if item.get("_direct"):
                score += 100
            if score < 0:
                continue
            key = urlsplit(item["url"])._replace(fragment="", query="").geturl().rstrip("/")
            if key not in unique or item.get("kind") == "page":
                unique[key] = {**item, "_score": score}
        return sorted(unique.values(), key=lambda item: -item["_score"])

    stage("Reading and comparing relevant pages…")
    read_candidates(ranked()[:6])
    relevant = ranked()
    # Second pass: a publisher/domain that matches the named subject gets a focused query.
    domains = list(plan["domains"])
    for item in relevant:
        host = urlsplit(item["url"]).hostname.removeprefix("www.")
        if plan["anchors"] & word_terms(host.replace(".", " ")) and host not in domains:
            domains.append(host)
    extra = []
    if domains:
        targeted = f"site:{domains[0]} {plan['query']}"
        if targeted not in queries:
            extra.append(targeted)
    if len(relevant) < 3:
        fallback = compact_query(plan["query"]) + (" official" if len(relevant) == 0 else "")
        if fallback not in queries:
            extra.append(fallback)
    if extra and time.monotonic() < deadline - 12:
        stage("Cross-checking with focused searches…")
        queries.extend(extra[:2])
        with ThreadPoolExecutor(max_workers=2) as pool:
            for results in pool.map(lookup, extra[:2]):
                candidates.extend(results)
        read_candidates(ranked()[:7])

    # Follow relevant links actually present on supplied/read pages, never guessed URLs.
    scanned = set()
    for _ in range(2):
        links = []
        for page in list(pages.values()):
            if page["url"] in scanned:
                continue
            scanned.add(page["url"])
            origin = urlsplit(page["url"]).hostname
            for link in page.get("_links", []):
                url = safe_url(link["url"])
                if not url or urlsplit(url).hostname != origin or url in attempted:
                    continue
                candidate = {"title": link["title"], "url": url, "snippet": link["title"]}
                if relevance(candidate, plan["anchors"], plan["query"]) > 5:
                    links.append(candidate)
        if not links or time.monotonic() >= deadline - 8:
            break
        stage("Checking linked publisher pages…")
        links.sort(key=lambda item: -relevance(item, plan["anchors"], plan["query"]))
        links = list({item["url"]: item for item in links}.values())[:2]
        read_candidates(links)
        candidates.extend(pages[link["url"]] for link in links if link["url"] in pages)

    final_candidates = ranked()
    requested_versions = re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", plan["question"])
    if plan["direct"]:
        requested_versions = re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", plan["query"])
    version_question = bool(requested_versions) and (
        plan["followup"]
        or plan["direct"]
        or re.search(r"version|release|dispo|availab", plan["question"], re.I)
    )
    if version_question:
        # Hard evidence gate: even a small model must never turn absent version evidence into "no".
        final_candidates = [
            item
            for item in final_candidates
            if any(
                mentions_version(item["title"] + " " + item["snippet"], version)
                for version in requested_versions
            )
        ]
        if not final_candidates and ranked():
            raise AppError(
                "Sources were found for the subject, but they do not verify the requested version ("
                + ", ".join(dict.fromkeys(requested_versions))
                + "). Its availability cannot be confirmed or ruled out. Provide an official release URL or retry the search.",
                502,
            )

    selected, domains_count = [], {}
    for item in final_candidates:
        host = urlsplit(item["url"]).hostname.removeprefix("www.")
        if domains_count.get(host, 0) >= 2 and not item.get("_direct"):
            continue
        domains_count[host] = domains_count.get(host, 0) + 1
        selected.append(item)
        if len(selected) == 5:
            break
    if not selected:
        raise AppError(
            "No relevant sources could be verified for this question. DuckDuckGo may be blocked or may have no matching results. Try the full subject name or provide an official page URL; no conclusion can be drawn from unrelated results.",
            502,
        )
    per_source = MAX_SNIPPET_WORDS // len(selected)
    sources = [
        {
            "id": index + 1,
            "title": item["title"][:240],
            "url": item["url"],
            "snippet": " ".join(item["snippet"].split()[:per_source]),
            "kind": item.get("kind", "snippet"),
            "published": item.get("published", ""),
        }
        for index, item in enumerate(selected)
    ]
    if all(source["kind"] == "snippet" for source in sources):
        warnings.append("Pages could not be read; this answer relies on search excerpts only.")
    if len(failures) == len(queries):
        warnings.append(
            "DuckDuckGo search was unavailable; this answer uses public pages read directly."
        )
    return sources, {
        "queries": queries,
        "pages_read": sum(source["kind"] == "page" for source in sources),
        "warnings": warnings,
    }
