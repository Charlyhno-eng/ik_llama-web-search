"""Bounded public-page reading and query-focused evidence extraction (stdlib only)."""

from html.parser import HTMLParser
import http.client
import ipaddress
import re
import socket
import ssl
import time
import unicodedata
from urllib.parse import urljoin, urlsplit, urlunsplit


STOP_WORDS = set(
    """a about ai all an and are as at avec avoir be been bien bon ce ces cet cette
comment comme dans de des dit do does du elle en est et eux faire for from how il ils in is it je la le
les leur lui ma mais me meme mes moi mon ne nous of on or ou par pas pour qu que quel quelle quels
quelles qui sa se ses son sur ta te tes the their them there these this to ton tu un une vos votre
vous was were what when where which who why with y yes model modele models modeles entreprise company
version versions dispo disponible disponibles available availability existe exist is there t has have
peux peut peux tu tell me please s'il plait quoi donc plus aussi non oui latest nouvelle nouveau
quand whether atil til yt itil sont bien merci information informations sujet site web internet
""".split()
)


def terms(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(
        r"\b(?:y-a-t-il|a-t-il|a-t-elle|t-il|t-elle|est-ce|qu-est-ce|peux-tu|pouvez-vous)\b",
        " ",
        normalized,
    )
    return [
        word
        for word in re.findall(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*", normalized)
        if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())
    ]


def word_terms(text):
    words = terms(text)
    # Versioned names in URLs (jev-1-14, qwen3.5) should still match their product name.
    words += [re.split(r"[-_.](?=\d)", word, maxsplit=1)[0] for word in words]
    return {word for word in words if not re.fullmatch(r"[\d.]+", word)}


def public_addresses(host, port):
    """Resolve once, reject every non-public destination, then pin the connection."""
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("No public address")
    for entry in addresses:
        address = ipaddress.ip_address(entry[4][0].split("%")[0])
        if not address.is_global or address.is_multicast or address.is_unspecified:
            raise ValueError("Only public Internet pages can be read")
    return addresses


def read_page(url, job=None, deadline=None):
    """No cookies, proxy, scripts, subresources, private addresses, or arbitrary ports."""
    deadline = min(deadline or float("inf"), time.monotonic() + 10)
    for _ in range(4):
        if job:
            job.check()
        if time.monotonic() >= deadline:
            raise TimeoutError("Page timeout")
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            raise ValueError("Invalid page URL")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if port not in (80, 443):
            raise ValueError("Only standard web ports can be read")
        addresses = public_addresses(parts.hostname, port)
        timeout = max(0.1, min(5, deadline - time.monotonic()))
        connection = http.client.HTTPConnection(parts.hostname, port, timeout=timeout)
        stream = None
        try:
            # Connect to the validated address, keeping the original Host and TLS SNI.
            last_error = None
            for family, kind, proto, _, address in addresses:
                if job:
                    job.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Page timeout")
                stream = socket.socket(family, kind, proto)
                stream.settimeout(max(0.1, min(timeout, deadline - time.monotonic())))
                if job:
                    job.track(stream)
                try:
                    stream.connect(address)
                    if parts.scheme == "https":
                        raw = stream
                        stream = ssl.create_default_context().wrap_socket(
                            raw, server_hostname=parts.hostname
                        )
                        if job:
                            job.untrack(raw)
                            job.track(stream)
                    break
                except OSError as exc:
                    last_error = exc
                    if job:
                        job.untrack(stream)
                    stream.close()
                    stream = None
            if stream is None:
                raise last_error or OSError("Connection failed")
            connection.sock = stream
            connection.request(
                "GET",
                urlunsplit(("", "", parts.path or "/", parts.query, "")),
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; LocalChat/1.0)",
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                url = urljoin(url, response.getheader("Location", ""))
                continue
            if response.status != 200:
                raise ValueError(f"Page returned HTTP {response.status}")
            kind = response.getheader("Content-Type", "").split(";")[0].lower()
            if kind not in ("text/html", "application/xhtml+xml", "text/plain"):
                raise ValueError("Unsupported page format")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected page encoding")
            chunks, size = [], 0
            while True:
                if job:
                    job.check()
                if response.isclosed():
                    break
                stream.settimeout(max(0.1, min(3, deadline - time.monotonic())))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Page timeout")
                data = response.read1(min(65536, 2_000_001 - size))
                if not data:
                    break
                chunks.append(data)
                size += len(data)
                if size > 2_000_000:
                    raise ValueError("Page too large")
            charset = response.headers.get_content_charset() or "utf-8"
            try:
                text = b"".join(chunks).decode(charset, errors="replace")
            except LookupError:
                text = b"".join(chunks).decode("utf-8", errors="replace")
            parser = PageParser(url)
            if kind == "text/plain":
                parser.blocks = [line.strip() for line in text.splitlines() if line.strip()]
            else:
                parser.feed(text)
                parser.close()
            return {
                "url": url,
                "title": parser.title,
                "blocks": parser.content(),
                "links": parser.links,
                "published": parser.published,
            }
        finally:
            if job and stream:
                job.untrack(stream)
            connection.close()
            if stream:
                stream.close()
    raise ValueError("Too many page redirects")


class PageParser(HTMLParser):
    """Prefer article/main blocks; discard navigation, scripts, forms and hidden text."""

    SKIP = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
        "aside",
        "template",
    }
    BLOCK = {"p", "div", "section", "article", "main", "li", "h1", "h2", "h3", "h4", "tr", "br"}
    VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.stack = []
        self.blocks, self.main_blocks, self.parts = [], [], []
        self.title = ""
        self.heading = ""
        self.published = ""
        self.links = []
        self.link = None

    def skipped(self):
        return any(skip for _, skip in self.stack)

    def flush(self):
        text = " ".join(" ".join(self.parts).split())
        self.parts = []
        if any(re.fullmatch(r"h[1-6]", tag) for tag, _ in self.stack) and text:
            self.heading = text
        elif text and self.heading and not text.startswith(self.heading):
            text = self.heading + ": " + text
        if len(text) >= 25:
            self.blocks.append(text)
            if any(tag in ("article", "main") for tag, _ in self.stack):
                self.main_blocks.append(text)

    def handle_starttag(self, tag, attributes):
        if len(self.stack) > 200:
            raise ValueError("Page markup is too deeply nested")
        attrs = dict(attributes)
        if tag in self.BLOCK:
            self.flush()
        if tag == "meta":
            name = attrs.get("property", attrs.get("name", "")).lower()
            if name in ("article:published_time", "datepublished", "date"):
                self.published = attrs.get("content", "")[:80]
            if name in ("og:title", "twitter:title") and not self.title:
                self.title = attrs.get("content", "")[:240]
        if tag == "time" and not self.published:
            self.published = attrs.get("datetime", "")[:80]
        if tag not in self.VOID:
            hidden = tag in self.SKIP or "hidden" in attrs or attrs.get("aria-hidden") == "true"
            self.stack.append((tag, hidden))
        if tag == "a" and not self.skipped():
            self.link = {"url": urljoin(self.url, attrs.get("href", "")), "title": ""}

    def handle_endtag(self, tag):
        if tag in self.BLOCK:
            self.flush()
        if tag == "a" and self.link:
            self.link["title"] = " ".join(self.link["title"].split())[:240]
            if self.link["title"]:
                self.links.append(self.link)
            self.link = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.skipped():
            return
        if any(tag == "title" for tag, _ in self.stack):
            self.title = (self.title + data)[:240]
            return
        self.parts.append(data)
        if self.link:
            self.link["title"] += " " + data

    def close(self):
        super().close()
        self.flush()

    def content(self):
        main = self.main_blocks
        # Some pages use <main> around an empty placeholder and render their text elsewhere.
        return list(dict.fromkeys(main if sum(map(len, main)) > 150 else self.blocks))[:1200]


def evidence_excerpt(blocks, query, max_words=160):
    """Select the best passages, including exact requested version/date numbers."""
    keys = set(terms(query))
    numbers = {term for term in keys if re.search(r"\d", term)}
    candidates = []
    seen = set()
    for block_index, block in enumerate(blocks):
        # Sliding windows prevent a long paragraph from hiding the relevant sentence at its end.
        words = block.split()
        for offset in range(0, len(words), 65):
            text = " ".join(words[offset : offset + 95])
            if len(text) < 25 or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            hits = keys & set(terms(text))
            number_hits = sum(
                bool(re.search(r"(?<![\w.])" + re.escape(number) + r"(?:\b|\.)", text, re.I))
                for number in numbers
            )
            release = bool(
                re.search(r"releas|publish|preview|availab|disponib|version", text, re.I)
            )
            score = (
                len(hits)
                + 6 * number_hits
                + (3 if number_hits and release else 0)
                + (0.2 if block_index < 3 else 0)
            )
            candidates.append((score, block_index, offset, text))
    chosen, used, count = [], set(), 0
    for score, index, offset, text in sorted(candidates, key=lambda item: -item[0]):
        if index in used or (chosen and score <= 0):
            continue
        words = text.split()[: max_words - count]
        if not words:
            break
        chosen.append((index, offset, " ".join(words)))
        used.add(index)
        count += len(words)
        if count >= max_words:
            break
    # Best evidence first: later context-budget trimming must preserve the strongest passage.
    return " … ".join(text for _, _, text in chosen)
