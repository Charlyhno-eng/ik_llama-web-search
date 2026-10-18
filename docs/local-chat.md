# Local Chat application

The application lives in `local_chat/` and the cross-platform `start.py` launcher. It owns an external `llama-server` process through loopback HTTP APIs. Inference code, kernels, C++ server and upstream interfaces remain unchanged.

## Startup and storage

`python start.py` starts the interface on Windows, Linux and macOS at `127.0.0.1:8080`, opens the browser and restores the saved model. Python 3.10+ is required; use `python3` or Windows `py -3` if that is your interpreter's command. No shell script or third-party Python package is required. The launcher also works with its full path from another working directory, including paths containing spaces. `python -m local_chat` is also available. The inference server receives a separate automatically assigned loopback port. With no saved model, Settings opens first. Files stay at their original paths; there is no model upload or copy.

The launcher checks `build/bin/llama-server`, then `build/bin/Release/llama-server`, with `.exe` on Windows. If neither exists, first model selection builds the CPU target with CMake and two parallel jobs (Release configuration for multi-configuration generators). CMake can discover installed Visual Studio compilers without `cl.exe` on PATH. Linux needs a C++20 compiler; macOS needs Xcode Command Line Tools and CMake. The existing upstream platform build defaults are preserved. `LLAMA_SERVER` can select another native executable; the app still owns its process rather than connecting to an independently running server. Builds and binaries are platform-specific; build separately on each OS.

Arguments retain the original settings: model, context, threads, `--jinja`, `--reasoning-tokens none`, and host `127.0.0.1`. One sequence is allowed; automatic context shifting is disabled because the app measures the prompt before generation. The template receives `enable_thinking=false`; support depends on the model/template.

Context sizes are 2,048, 4,096, 6,144 and 8,192 tokens (default 4,096). Generation CPU threads accept 1–256 (default four). Prompt CPU threads accept 0–256 (default zero, which follows generation threads); a positive value is passed independently as `--threads-batch`. Temperature accepts finite numbers from 0–2 (default 0.5). Maximum response length accepts integers from 128–4,096 tokens (default 1,024). All controls are in Settings and saved together with the model path. Missing or invalid saved generation values fall back to the shared defaults in `local_chat/config.py`, so older settings migrate automatically. Local changes are written only to the Git-ignored `.local-chat/settings.json`, never to the shared defaults. Model selection checks `.gguf`, file existence, magic bytes and GGUF version 2/3. This does not guarantee model architecture support; the engine log explains loading failures.

Settings are written atomically to `.local-chat/settings.json`. Engine and build logs are replaced on each load/build. `LOCAL_CHAT_DATA_DIR` overrides this directory. Moving or deleting a model requires selecting its new location. Ctrl+C cancels active work and stops owned processes. Linux/macOS use an isolated process group with SIGTERM and a SIGKILL fallback; Windows uses a new process group and `taskkill /T /F` to stop the owned tree, including compiler processes. Closing the browser alone does not stop the application.

## Contextual web retrieval

Web search is optional. Each new request retrieves evidence for the current question:

1. A deterministic query builder identifies short follow-ups and recovers the subject from earlier **user questions**, avoiding earlier assistant assertions. Exact requested version numbers are retained. New named subjects start a new topic.
2. Follow-ups with a named prior topic or relevant prior domain use the deterministic query directly. Otherwise, the local model can propose a concise standalone query, with a four-second deadline and a 48-token limit. The proposal is rejected if it drops the subject/version, adds URLs, or returns commentary. A deterministic query remains available if rewriting fails. The planner is cancellable and does not modify inference code.
3. DuckDuckGo Lite then HTML are queried with a global region, using the contextual query and, when useful, a focused domain query. A second pass can broaden/reformulate a poor result set or focus on a publisher domain matching the subject. Additional searches and linked-page exploration stop once four relevant, readable sources satisfy the evidence checks. At most four search queries are used. Each endpoint has a six-second timeout; no CAPTCHA is bypassed. Successful result sets have a bounded, three-minute, in-memory cache to reduce repeated requests.
4. Provided HTTP(S) URLs are read directly, independent of search indexing. When all provided pages are readable and satisfy requested-version checks, a page-reading question skips DuckDuckGo; explicit comparisons/cross-checks still search. Relevant earlier source URLs can also be read again; old excerpts are **never reused as fresh evidence**. Prior domains are hints, not proof that a site is official.
5. Results are scored against subject words, titles, domains, excerpts and the requested version. Matches on a number alone do not count as topic matches. Unrelated results are discarded. Relevant pages are read in parallel (three workers), then actual relevant links on the same host can be followed up to two levels (two pages per level). No article URL is invented.
6. Page reading extracts main/article text while dropping navigation, scripts, hidden text, forms and footers. Query-focused passages prioritize the requested version even when it occurs late in a page. The strongest evidence comes first so budget trimming preserves it. Unreadable pages retain their search excerpts and are labeled accordingly.
7. Duplicate URLs are removed. Up to four sources are selected, with no more than two from the same hostname except explicit user URLs. Fewer sources are returned when relevance is limited. Evidence is capped at 600 words total; there is no padding with unrelated results.

The retrieval phase has a shared 40-second budget including the optional local planner. Network operations use the remaining budget; operating-system DNS resolution can still overrun it. Prior source reads overlap independent search requests. The first ranked batch contains at most four pages; fallback reads can inspect more candidates when initial evidence is insufficient. Each public page has a ten-second deadline, at most three redirects and a two-megabyte body limit. Pages are read as HTML/plain text; JavaScript, PDFs, logins and paywalls are not executed or bypassed. Search can be unavailable even when supplied/previous page URLs remain readable. The interface shows the actual queries, evidence type and relevant limitations. If no relevant evidence is available, generation stops with a search error rather than giving the model unrelated snippets. For a question about a specific version, at least one selected source must mention that version: otherwise the app reports that availability cannot be confirmed or ruled out, without asking the model to guess.

Queries can contain subject names derived from earlier messages. DuckDuckGo receives these queries; the local conversation is not uploaded wholesale. Destination websites receive ordinary HTTP page requests. Reading pages expands network activity beyond the search provider.

## Generation, evidence and performance

The prompt includes today's UTC date and instructs the model to answer in the user's language, prefer current evidence over earlier assistant claims, and distinguish publisher claims from demonstrated facts. French/English language cues are made explicit to help small models answer in French even when pages are English. Missing evidence must not be treated as proof that a product/version does not exist. These instructions improve grounding but cannot guarantee the model follows them.

Sources are untrusted data, never tool instructions. Metadata distinguishes a passage actually read from a page from a search-engine excerpt; an available publication timestamp is retained. URLs remain application metadata. Only citations `[n]` matching this answer's source IDs become clickable. Free-form URLs/Markdown links written by the model are not activated. A valid link does not by itself prove the cited claim is supported.

`/apply-template` and `/tokenize` count the real prompt. The response reserve is the selected maximum response length, capped at half the active context, plus a 64-token safety margin. Increasing it can omit more history, shorten more sources or reject a question that cannot fit. Oldest whole turns are omitted first using a binary search with exact template/tokenizer measurements, then source passages shortened. The current question is never silently cut. Visible history remains intact; the interface indicates context omissions. The selected temperature is sent with the final answer request; the search query planner keeps its independent temperature of zero and 48-token output limit.

Time to first token uses a monotonic clock from request handling to the first nonempty generated text, including retrieval, planning and context preparation. A tooltip reports the delay from the final model request to the first text. Empty role chunks do not count. Generation speed and token count use the engine's final `timings`/`usage` fields, not the number of SSE chunks. If timing is absent, an estimate uses completion tokens (or the real tokenizer) and streaming duration, displayed with `≈`. Metrics persist with the conversation; older conversations without measurements show none.

## Interface and access controls

English HTML/CSS/JavaScript, no frontend build, CDN, external font or runtime packages. Navy is the default theme; a light theme is available. Enter sends, Shift+Enter inserts a line, Ctrl/Cmd+K opens a new chat. Browser storage retains up to 40 conversations and 80 complete exchanges each. Legacy storage keys migrate transparently so the rename preserves existing history and theme preferences.

The server binds only to `127.0.0.1`. APIs require a per-launch random token; Host and Origin are checked. The file browser lists readable folders and `.gguf` files, with no arbitrary file-read API. Paths are passed as subprocess arguments without a shell. Chat text uses safe DOM nodes rather than interpreting model HTML. Completed message nodes are reused during streaming; only the active response is rebuilt, at most about every 80 ms, with its first text rendered immediately.

Public-page requests reject credentials, non-HTTP(S) schemes, nonstandard ports and all private/loopback/link-local/multicast destinations. DNS results are validated and connections are pinned to the validated address while retaining TLS hostname verification. Redirect targets are validated again. Environment proxies are not used for page reads. Cancellation shuts down owned sockets, including local query rewriting and active public-page connections; an in-progress DuckDuckGo request may finish its current timeout.

## Validation

Offline automated checks:

```sh
python -m unittest discover -s local_chat/tests -v
node --check local_chat/static/app.js
```

Coverage includes settings/GGUF validation, DuckDuckGo parsing, follow-up resolution, topic changes, version preservation, irrelevant-result rejection, direct URL reading, refreshed prior sources, excerpt budgets, failed-page fallback, public-IP and redirect checks, chat token budgets, HTTP protection, streaming metrics and cancellation. No model download or network is required for this suite. Node is only needed for the optional JavaScript syntax check/development.

Manual checks use Chrome on desktop/mobile and a real local GGUF: initial model selection, ordinary French/English replies, the Jev/TypeSafe follow-up case, direct release URLs, source-link provenance, timing display, stopping, persistence, settings and both themes. The live Jev/TypeSafe 1.14 follow-up recovered the official release page by revisiting the publisher site, even while DuckDuckGo was presenting a CAPTCHA. Qwen3.5 2B answered in French and reported real engine timings, but could still misinterpret alias changes or relative dates in its summary. Better retrieval does not eliminate a small model’s reasoning errors. Deterministic fixtures keep the relevance regression reproducible despite provider blocking.

Advanced upstream build instructions are available in the [ik_llama.cpp documentation](https://github.com/ikawrakow/ik_llama.cpp#readme).
