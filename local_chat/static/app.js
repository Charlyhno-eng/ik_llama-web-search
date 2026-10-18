"use strict";

const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="local-chat-token"]').content;
const icons = {
    settings:
        '<path d="m9 3 1-1h4l1 1 .5 2 2 .8 1.8-.6 2 3.5-1.3 1.5v2.6l1.3 1.5-2 3.5-1.8-.6-2 .8-.5 2-1 1h-4l-1-1-.5-2-2-.8-1.8.6-2-3.5L4 12.4V9.8L2.7 8.3l2-3.5 1.8.6 2-.8Z"/><circle cx="12" cy="11" r="3"/>',
    moon: '<path d="M20.8 13A9 9 0 0 1 11 3.2 9 9 0 1 0 20.8 13Z"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a19 19 0 0 1 0 18 19 19 0 0 1 0-18Z"/>',
    spark: '<path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3ZM20 2v4m-2-2h4"/>',
    edit: '<path d="m14 5 5 5M4 20l5-1L21 7a2 2 0 0 0-5-5L4 14v6ZM12 20h9"/>',
    "arrow-up": '<path d="M12 19V5m-6 6 6-6 6 6"/>',
    chevron: '<path d="m7 10 5 5 5-5"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    file: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6Zm0 0v6h6M8 14h8M8 17h5"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    home: '<path d="m3 10 9-7 9 7M5 9v12h5v-7h4v7h5V9"/>',
    up: '<path d="m6 10 6-6 6 6M12 4v16"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
    trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
    copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
};

function icon(name) {
    const span = document.createElement("span");
    span.dataset.icon = name;
    // These are compile-time SVG paths, never model or user input.
    span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.file}</svg>`;
    return span;
}
document
    .querySelectorAll("[data-icon]")
    .forEach((node) => node.replaceWith(icon(node.dataset.icon)));

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

let toastTimeout;
function toast(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        $("toast").hidden = true;
    }, 4500);
}

async function api(path, data) {
    const response = await fetch(path, {
        method: data === undefined ? "GET" : "POST",
        headers: {
            "Content-Type": "application/json",
            "X-Local-Chat-Token": token,
        },
        body: data === undefined ? undefined : JSON.stringify(data),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "The request failed.");
    return value;
}

const STORAGE_KEY = "local-chat.v1";
let conversations = [];
let currentId = null;
let status = null;
let busy = false;
let web = false;
let currentRequest = null;
let stopping = false;
let folder = null;
let booted = false;
let lastState = null;
let scrollQueued = false;
let storageWarning = false;
function persist() {
    try {
        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify({ conversations, currentId }),
        );
        localStorage.removeItem("ik-local-chat.v1");
    } catch {
        if (!storageWarning)
            toast(
                "Browser storage is full or unavailable. This conversation is open but will not be saved.",
            );
        storageWarning = true;
    }
}

try {
    const saved = JSON.parse(
        localStorage.getItem(STORAGE_KEY) ||
            localStorage.getItem("ik-local-chat.v1") ||
            "null",
    );
    if (saved && Array.isArray(saved.conversations)) {
        conversations = saved.conversations
            .filter(
                (chat) =>
                    typeof chat.id === "string" &&
                    typeof chat.title === "string" &&
                    Array.isArray(chat.messages),
            )
            .slice(0, 40);
        for (const chat of conversations) {
            chat.messages = chat.messages
                .filter(
                    (message) =>
                        ["user", "assistant"].includes(message.role) &&
                        typeof message.content === "string",
                )
                .slice(-160)
                .map((message) => Object.fromEntries(
                    Object.entries(message).filter(([key]) => [
                        "role", "content", "sources", "pending", "note", "error",
                        "budget", "metrics", "search",
                    ].includes(key)),
                ));
            for (const message of chat.messages) {
                if (message.budget && typeof message.budget === "object") {
                    message.budget = Object.fromEntries(
                        Object.entries(message.budget).filter(([key]) => [
                            "prompt_tokens", "response_tokens", "context_size",
                            "removed_turns", "shortened_sources", "snippet_words",
                        ].includes(key)),
                    );
                }
                if (message.pending) {
                    message.pending = false;
                    message.note = "Response interrupted when the page closed.";
                }
                if (!Array.isArray(message.sources)) message.sources = [];
            }
        }
        currentId = saved.currentId;
        persist();
    }
} catch {
    /* A corrupt or unavailable browser store must not prevent chatting. */
}

function currentChat() {
    return conversations.find((chat) => chat.id === currentId);
}

function newChat() {
    if (busy) return;
    currentId = null;
    persist();
    render();
    closeSidebar();
    $("prompt").focus();
}

function closeSidebar() {
    $("sidebar").classList.remove("open");
    $("sidebar-backdrop").hidden = true;
}

function renderHistory() {
    $("history").replaceChildren();
    if (!conversations.length) {
        $("history").append(
            element(
                "p",
                "history-empty",
                "Your conversations will appear here.",
            ),
        );
        return;
    }
    for (const chat of conversations) {
        const row = element(
            "div",
            `history-item${chat.id === currentId ? " active" : ""}`,
        );
        const select = element("button", "history-select", chat.title);
        select.title = chat.title;
        select.disabled = busy;
        if (chat.id === currentId) select.setAttribute("aria-current", "true");
        select.onclick = () => {
            currentId = chat.id;
            persist();
            render();
            closeSidebar();
        };
        const remove = element("button", "history-delete");
        remove.append(icon("trash"));
        remove.setAttribute("aria-label", `Delete “${chat.title}”`);
        remove.title = "Delete conversation";
        remove.disabled = busy;
        remove.onclick = () => {
            if (!window.confirm("Delete this conversation from this device?"))
                return;
            conversations = conversations.filter((item) => item.id !== chat.id);
            if (currentId === chat.id) currentId = null;
            persist();
            render();
        };
        row.append(select, remove);
        $("history").append(row);
    }
}

function trustedURL(value) {
    try {
        const url = new URL(value);
        return ["https:", "http:"].includes(url.protocol) &&
            !url.username &&
            !url.password
            ? url.href
            : null;
    } catch {
        return null;
    }
}

function sourceLink(source, text, className) {
    const url = trustedURL(source?.url);
    if (!url) return document.createTextNode(text);
    const a = element("a", className, text);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = `${source.title}\n${url}`;
    return a;
}

function inlineText(parent, text, sources) {
    // No HTML interpretation, arbitrary Markdown links, or model-generated hrefs.
    const pattern = /(\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[(\d{1,2})\])/g;
    let cursor = 0;
    for (const match of text.matchAll(pattern)) {
        parent.append(document.createTextNode(text.slice(cursor, match.index)));
        if (match[2]) parent.append(element("strong", "", match[2]));
        else if (match[3]) parent.append(element("code", "", match[3]));
        else {
            const source = sources.find((item) => item.id === Number(match[4]));
            parent.append(
                source
                    ? sourceLink(source, match[0], "citation")
                    : document.createTextNode(match[0]),
            );
        }
        cursor = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(cursor)));
}

function markdown(text, sources) {
    const root = element("div", "message-body");
    // Parse fences line by line so incomplete streamed code remains readable.
    let code = null;
    let paragraph = [];
    let list = null;
    const flush = () => {
        if (paragraph.length) {
            const p = element("p");
            inlineText(p, paragraph.join("\n"), sources);
            root.append(p);
            paragraph = [];
        }
        list = null;
    };
    const lines = text.split("\n");
    const tableCells = (line) =>
        line
            .trim()
            .replace(/^\||\|$/g, "")
            .split(/(?<!\\)\|/)
            .map((cell) => cell.trim().replace(/\\\|/g, "|"));
    for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        if (line.trimStart().startsWith("```")) {
            flush();
            if (code) code = null;
            else {
                const pre = element("pre");
                code = element("code");
                pre.append(code);
                root.append(pre);
            }
        } else if (code) {
            code.append(document.createTextNode(line + "\n"));
        } else if (!line.trim()) {
            flush();
        } else if (
            line.includes("|") &&
            /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(
                lines[index + 1] || "",
            )
        ) {
            flush();
            const wrapper = element("div", "table-scroll");
            const table = element("table");
            const head = element("thead");
            const row = element("tr");
            const headings = tableCells(line);
            for (const value of headings) {
                const cell = element("th");
                cell.scope = "col";
                inlineText(cell, value, sources);
                row.append(cell);
            }
            head.append(row);
            table.append(head);
            const body = element("tbody");
            index++;
            while (lines[index + 1]?.includes("|") && lines[index + 1].trim()) {
                const values = tableCells(lines[++index]);
                const row = element("tr");
                headings.forEach((_, cellIndex) => {
                    const cell = element("td");
                    inlineText(cell, values[cellIndex] || "", sources);
                    row.append(cell);
                });
                body.append(row);
            }
            table.append(body);
            wrapper.append(table);
            root.append(wrapper);
        } else if (/^#{1,6}\s/.test(line)) {
            flush();
            const heading = element("h3");
            inlineText(heading, line.replace(/^#{1,6}\s+/, ""), sources);
            root.append(heading);
        } else if (/^\s*([-*]|\d+[.)])\s/.test(line)) {
            if (paragraph.length) flush();
            const kind = /^\s*\d/.test(line) ? "ol" : "ul";
            if (!list || list.tagName.toLowerCase() !== kind) {
                list = element(kind);
                root.append(list);
            }
            const li = element("li");
            inlineText(li, line.replace(/^\s*([-*]|\d+[.)])\s+/, ""), sources);
            list.append(li);
        } else {
            list = null;
            paragraph.push(line);
        }
    }
    flush();
    return root;
}

let renderedChat = null;
const messageNodes = new WeakMap();
let lastMessageRender = 0;

function renderMessages() {
    const chat = currentChat();
    $("welcome").hidden = !!chat?.messages.length;
    const container = $("messages");
    if (renderedChat !== chat) {
        container.replaceChildren();
        renderedChat = chat;
    }
    if (!chat) return;
    // Storage trims the oldest pairs after 80 exchanges; drop their old DOM nodes.
    if (container.firstChild && messageNodes.get(chat.messages[0])?.node !== container.firstChild)
        container.replaceChildren();
    for (const message of chat.messages) {
        const previous = messageNodes.get(message);
        if (previous && !previous.pending && !message.pending) {
            if (previous.node.parentNode !== container) container.append(previous.node);
            continue;
        }
        const article = element("article", `message ${message.role}`);
        if (message.role === "user") {
            article.append(element("div", "user-bubble", message.content));
        } else {
            const label = element("div", "assistant-label");
            label.append(
                element("span", "assistant-avatar"),
                document.createTextNode("Assistant"),
            );
            label.firstChild.append(icon("spark"));
            article.append(label);
            const sources = Array.isArray(message.sources)
                ? message.sources
                : [];
            if (message.content)
                article.append(markdown(message.content, sources));
            else if (message.pending) {
                const dots = element("div", "thinking-placeholder");
                dots.setAttribute("aria-label", "Preparing a response");
                dots.append(element("span"), element("span"), element("span"));
                article.append(dots);
            }
            if (sources.length) {
                const pagesRead = sources.filter(
                    (source) => source.kind === "page",
                ).length;
                const details = element("details", "sources");
                details.append(
                    element(
                        "summary",
                        "",
                        `${sources.length} source${sources.length === 1 ? "" : "s"} · ${pagesRead ? `${pagesRead} page${pagesRead === 1 ? "" : "s"} read` : "search excerpts"}`,
                    ),
                );
                const grid = element("div", "source-grid");
                for (const source of sources) {
                    if (!trustedURL(source.url)) continue;
                    const card = element("div", "source-card");
                    card.append(
                        sourceLink(source, `[${source.id}] ${source.title}`),
                    );
                    card.append(
                        element(
                            "span",
                            "source-domain",
                            new URL(source.url).hostname,
                        ),
                    );
                    card.append(element("p", "", source.snippet));
                    card.append(
                        element(
                            "small",
                            "source-kind",
                            `${source.kind === "page" ? "Page excerpt" : "Search excerpt"}${source.published ? " · " + source.published.slice(0, 10) : ""}`,
                        ),
                    );
                    grid.append(card);
                }
                details.append(grid);
                if (message.search?.queries?.length) {
                    const queries = element("div", "search-queries");
                    queries.append(element("strong", "", "Search queries"));
                    for (const query of message.search.queries)
                        queries.append(element("div", "", query));
                    details.append(queries);
                }
                article.append(details);
            }
            for (const warning of message.search?.warnings || [])
                article.append(element("p", "message-note", warning));
            if (message.note)
                article.append(
                    element(
                        "p",
                        `message-note${message.error ? " message-error" : ""}`,
                        message.note,
                    ),
                );
            if (message.budget && !message.pending) {
                const notes = [];
                if (message.budget.removed_turns)
                    notes.push(
                        `${message.budget.removed_turns} earlier turn(s) omitted from context`,
                    );
                if (message.budget.shortened_sources)
                    notes.push("source excerpts shortened to fit the context");
                if (notes.length)
                    article.append(
                        element("p", "message-note", notes.join(" · ")),
                    );
            }
            if (message.content && !message.pending) {
                const actions = element("div", "message-actions");
                const copy = element("button", "icon-button");
                copy.append(icon("copy"));
                copy.title = "Copy response";
                copy.setAttribute("aria-label", "Copy response");
                copy.onclick = async () => {
                    try {
                        await navigator.clipboard.writeText(message.content);
                        toast("Response copied.");
                    } catch {
                        toast("Copying is unavailable in this browser.");
                    }
                };
                actions.append(copy);
                const metrics = message.metrics;
                if (metrics && Number.isFinite(metrics.first_token_ms)) {
                    const stats = element("div", "response-metrics");
                    if (
                        Number.isFinite(metrics.tokens_per_second) &&
                        metrics.tokens_per_second > 0
                    ) {
                        const speed = element(
                            "span",
                            "",
                            `${metrics.engine_timing ? "" : "≈ "}${metrics.tokens_per_second.toFixed(1)} tok/s`,
                        );
                        speed.title = metrics.engine_timing
                            ? "Generation speed reported by the local engine"
                            : "Estimated from the token count and streaming duration";
                        stats.append(speed);
                    }
                    const first = element(
                        "span",
                        "",
                        `First token ${(metrics.first_token_ms / 1000).toFixed(2)} s`,
                    );
                    first.title = `From sending the message, including search and context preparation. Model request to first text: ${(metrics.model_first_token_ms / 1000).toFixed(2)} s.`;
                    stats.append(first);
                    actions.append(stats);
                }
                article.append(actions);
            }
        }
        if (previous?.node.parentNode === container) previous.node.replaceWith(article);
        else container.append(article);
        messageNodes.set(message, { node: article, pending: !!message.pending });
    }
    lastMessageRender = performance.now();
}

function render() {
    renderHistory();
    renderMessages();
    updateControls();
}

function scheduleMessages() {
    if (scrollQueued) return;
    scrollQueued = true;
    const chat = currentChat();
    const active = chat?.messages.at(-1);
    const firstText = active?.content && !messageNodes.get(active)?.node.querySelector(".message-body");
    const delay = firstText ? 0 : Math.max(0, 80 - (performance.now() - lastMessageRender));
    setTimeout(() => requestAnimationFrame(() => {
        const scroll = $("chat-scroll");
        const atBottom =
            scroll.scrollHeight - scroll.clientHeight - scroll.scrollTop < 130;
        renderMessages();
        if (atBottom) scroll.scrollTop = scroll.scrollHeight;
        scrollQueued = false;
    }), delay);
}

function updateControls() {
    $("send").hidden = busy;
    $("stop").hidden = !busy;
    $("stop").disabled = stopping;
    $("send").disabled = status?.state !== "ready" || !$("prompt").value.trim();
    $("new-chat").disabled = busy;
    $("web-toggle").disabled = busy;
    $("load-model").disabled =
        busy || ["loading", "building"].includes(status?.state);
    $("messages").setAttribute("aria-busy", String(busy));
}

function setWeb(value) {
    web = value;
    $("web-toggle").setAttribute("aria-pressed", String(web));
    $("composer-note").textContent = web
        ? "A contextual search query is sent to DuckDuckGo. Relevant public pages are read; your model writes the answer locally."
        : "Responses can contain mistakes. Check important information.";
}

async function refreshStatus() {
    try {
        status = await api("/api/status");
        const active = status.active;
        const model = active?.model_path.split(/[\\/]/).pop();
        $("top-model").textContent = model || "Local Chat";
        $("engine-model").textContent =
            model ||
            status.settings.model_path.split(/[\\/]/).pop() ||
            "No model loaded";
        $("engine-model").title =
            active?.model_path || status.settings.model_path;
        $("engine-detail").textContent = active
            ? `${active.context_size.toLocaleString("en-US")} tokens · CPU threads: ${active.threads} generation, ${active.threads_batch || active.threads} prompt`
            : {
                  idle: "Choose a GGUF to get started",
                  loading: "Loading model…",
                  building: "Building server…",
                  error: "Model needs attention",
              }[status.state] || "";
        $("engine-dot").className = `status-dot ${status.state}`;
        $("status-banner").hidden = status.state === "ready" && !status.warning;
        $("status-banner").className =
            `status-banner${status.state === "error" ? " error" : ""}`;
        $("status-banner").textContent = status.warning || status.message;
        if (
            status.state === "ready" &&
            ["loading", "building"].includes(lastState)
        )
            toast("Your model is ready.");
        if (!booted) {
            booted = true;
            if (!status.settings.model_path) openSettings();
        }
        lastState = status.state;
        updateControls();
    } catch (error) {
        status = null;
        $("status-banner").hidden = false;
        $("status-banner").className = "status-banner error";
        $("status-banner").textContent =
            "Disconnected from the application. Run python start.py again, then reload this page.";
        $("engine-dot").className = "status-dot error";
        $("engine-detail").textContent = "Application disconnected";
        updateControls();
    }
}

function openSettings() {
    if (!status) {
        toast("The application is unreachable.");
        return;
    }
    $("model-path").value = status.settings.model_path;
    $("context-size").value = status.settings.context_size;
    $("threads").value = status.settings.threads;
    $("threads-batch").value = status.settings.threads_batch ?? 0;
    $("temperature").value = status.settings.temperature;
    $("max-response-tokens").value = status.settings.max_response_tokens;
    updateContext();
    $("settings-error").hidden = true;
    if (!$("settings-dialog").open) $("settings-dialog").showModal();
    closeSidebar();
}

function updateContext() {
    $("context-value").textContent =
        `${Number($("context-size").value).toLocaleString("en-US")} tokens`;
}

let browseSequence = 0;
async function browseFolder(path) {
    const sequence = ++browseSequence;
    $("folder-path").disabled = true;
    $("file-list").replaceChildren(
        element("p", "file-empty", "Reading folder…"),
    );
    try {
        const result = await api(
            "/api/files" +
                (path ? `?directory=${encodeURIComponent(path)}` : ""),
        );
        if (sequence !== browseSequence) return;
        folder = result;
        $("folder-path").value = result.directory;
        $("folder-up").disabled = result.parent === result.directory;
        $("file-list").replaceChildren();
        const files = result.entries.filter((entry) => !entry.directory).length;
        $("file-count").textContent =
            `${files} GGUF file${files === 1 ? "" : "s"} · folders and .gguf only`;
        for (const entry of result.entries) {
            const button = element("button", "file-entry");
            button.type = "button";
            button.append(
                icon(entry.directory ? "folder" : "file"),
                element("span", "filename", entry.name),
            );
            button.append(
                element(
                    "span",
                    "filesize",
                    entry.directory ? "Folder" : formatSize(entry.size),
                ),
            );
            button.onclick = () => {
                if (entry.directory) browseFolder(entry.path);
                else {
                    $("model-path").value = entry.path;
                    $("files-dialog").close();
                }
            };
            $("file-list").append(button);
        }
        if (!result.entries.length)
            $("file-list").append(
                element(
                    "p",
                    "file-empty",
                    "No folders or GGUF files here. Open another folder.",
                ),
            );
    } catch (error) {
        if (sequence !== browseSequence) return;
        $("file-list").replaceChildren(
            element("p", "file-empty message-error", error.message),
        );
    } finally {
        if (sequence === browseSequence) $("folder-path").disabled = false;
    }
}

function formatSize(bytes) {
    return bytes >= 1024 ** 3
        ? `${(bytes / 1024 ** 3).toFixed(2)} GB`
        : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function conversationPayload(chat) {
    const output = [];
    // Failed/empty assistant replies do not create invalid consecutive user turns.
    for (let index = 0; index < chat.messages.length - 2; index++) {
        const user = chat.messages[index];
        const assistant = chat.messages[index + 1];
        if (
            user.role === "user" &&
            assistant?.role === "assistant" &&
            assistant.content.trim() &&
            !assistant.error
        ) {
            output.push(
                { role: "user", content: user.content },
                { role: "assistant", content: assistant.content },
            );
            index++;
        }
    }
    return output
        .slice(-80)
        .concat({ role: "user", content: chat.messages.at(-2).content });
}

async function sendMessage(event) {
    event.preventDefault();
    const question = $("prompt").value.trim();
    if (!question || busy) return;
    if (status?.state !== "ready") {
        openSettings();
        return;
    }
    let chat = currentChat();
    if (!chat) {
        chat = {
            id: crypto.randomUUID(),
            title: question.slice(0, 65),
            messages: [],
        };
        conversations.unshift(chat);
        conversations = conversations.slice(0, 40);
        currentId = chat.id;
    }
    const assistant = {
        role: "assistant",
        content: "",
        sources: [],
        pending: true,
    };
    chat.messages.push({ role: "user", content: question }, assistant);
    // Bound browser storage while keeping complete pairs.
    if (chat.messages.length > 160) chat.messages = chat.messages.slice(-160);
    busy = true;
    stopping = false;
    currentRequest = crypto.randomUUID();
    $("prompt").value = "";
    resizePrompt();
    $("activity").textContent = web
        ? "Searching DuckDuckGo…"
        : "Preparing a response…";
    render();
    persist();
    $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
    let completed = false;
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Local-Chat-Token": token,
            },
            body: JSON.stringify({
                messages: conversationPayload(chat),
                previous_sources:
                    [...chat.messages.slice(0, -2)]
                        .reverse()
                        .find(
                            (message) =>
                                message.role === "assistant" &&
                                !message.error &&
                                message.sources?.length,
                        )
                        ?.sources.slice(0, 5)
                        .map((source) => ({
                            title: source.title,
                            url: source.url,
                        })) || [],
                web,
                id: currentRequest,
            }),
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder
                .decode(value, { stream: true })
                .replace(/\r\n/g, "\n");
            let end;
            while ((end = buffer.indexOf("\n\n")) >= 0) {
                const block = buffer.slice(0, end);
                buffer = buffer.slice(end + 2);
                const kind = block.match(/^event: (.+)$/m)?.[1];
                const raw = block.match(/^data: (.+)$/m)?.[1];
                if (!raw) continue;
                const data = JSON.parse(raw);
                if (kind === "stage") $("activity").textContent = data.message;
                if (kind === "meta") {
                    assistant.sources = data.sources;
                    assistant.budget = data.budget;
                    assistant.search = data.search;
                    $("activity").textContent = "Writing a response…";
                }
                if (kind === "token") {
                    const firstText = !assistant.content;
                    assistant.content += data.text;
                    if (firstText) renderMessages();
                }
                if (kind === "metrics") assistant.metrics = data;
                if (kind === "error") throw new Error(data.message);
                if (kind === "cancelled") {
                    assistant.note = "Response stopped.";
                    completed = true;
                }
                if (kind === "done") {
                    completed = true;
                    if (data.metrics) assistant.metrics = data.metrics;
                    if (data.limited)
                        assistant.note =
                            "Response limit reached. You can ask the model to continue.";
                }
                scheduleMessages();
            }
        }
        if (!completed)
            throw new Error(
                "The connection closed before the response finished.",
            );
    } catch (error) {
        assistant.error = !stopping;
        assistant.note = stopping ? "Response stopped." : error.message;
        if (!assistant.content && !stopping) {
            $("prompt").value = question;
        }
    } finally {
        assistant.pending = false;
        busy = false;
        stopping = false;
        currentRequest = null;
        $("activity").textContent = "";
        render();
        persist();
        resizePrompt();
        $("prompt").focus();
        refreshStatus();
    }
}

async function stopMessage() {
    if (!currentRequest || stopping) return;
    stopping = true;
    $("activity").textContent = "Stopping the response…";
    updateControls();
    try {
        await api("/api/cancel", { id: currentRequest });
    } catch (error) {
        toast(error.message);
        stopping = false;
        updateControls();
    }
}

function resizePrompt() {
    $("prompt").style.height = "auto";
    $("prompt").style.height = `${Math.min($("prompt").scrollHeight, 160)}px`;
    updateControls();
}

$("new-chat").onclick = newChat;
$("settings-open").onclick = openSettings;
$("model-button").onclick = openSettings;
$("status-banner").onclick = openSettings;
$("context-size").oninput = updateContext;
$("web-toggle").onclick = () => setWeb(!web);
$("composer").onsubmit = sendMessage;
$("stop").onclick = stopMessage;
$("prompt").oninput = resizePrompt;
$("prompt").onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!busy) $("composer").requestSubmit();
    }
};
document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        newChat();
    }
});
$("menu-toggle").onclick = () => {
    $("sidebar").classList.add("open");
    $("sidebar-backdrop").hidden = false;
};
$("sidebar-backdrop").onclick = closeSidebar;
document.querySelectorAll("[data-close]").forEach((button) => {
    button.onclick = () => $(button.dataset.close).close();
});
document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
        const rect = dialog.getBoundingClientRect();
        if (
            event.target === dialog &&
            (event.clientX < rect.left ||
                event.clientX > rect.right ||
                event.clientY < rect.top ||
                event.clientY > rect.bottom)
        )
            dialog.close();
    });
});
document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.onclick = () => {
        if (busy) return;
        $("prompt").value = button.dataset.prompt;
        setWeb(button.dataset.web === "true");
        resizePrompt();
        $("prompt").focus();
    };
});
$("browse-open").onclick = () => {
    $("files-dialog").showModal();
    const path = $("model-path").value;
    const separator = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    browseFolder(separator >= 0 ? path.slice(0, separator + 1) : null);
};
$("folder-form").onsubmit = (event) => {
    event.preventDefault();
    browseFolder($("folder-path").value);
};
$("folder-home").onclick = () => browseFolder(folder?.home);
$("folder-up").onclick = () => browseFolder(folder?.parent);
$("settings-form").onsubmit = async (event) => {
    event.preventDefault();
    $("settings-error").hidden = true;
    $("load-model").disabled = true;
    try {
        await api("/api/settings", {
            model_path: $("model-path").value,
            context_size: Number($("context-size").value),
            threads: Number($("threads").value),
            threads_batch: Number($("threads-batch").value),
            temperature: Number($("temperature").value),
            max_response_tokens: Number($("max-response-tokens").value),
        });
        $("settings-dialog").close();
        await refreshStatus();
    } catch (error) {
        $("settings-error").textContent = error.message;
        $("settings-error").hidden = false;
    } finally {
        updateControls();
    }
};
$("logs-details").ontoggle = async () => {
    if (!$("logs-details").open) return;
    try {
        $("logs").textContent = (await api("/api/logs")).text;
    } catch (error) {
        $("logs").textContent = error.message;
    }
};

try {
    const savedTheme =
        localStorage.getItem("local-chat.theme") ||
        localStorage.getItem("ik-local-chat.theme");
    document.documentElement.dataset.theme = savedTheme || "dark";
} catch {
    document.documentElement.dataset.theme = "dark";
}
$("theme-toggle").onclick = () => {
    const theme =
        document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try {
        localStorage.setItem("local-chat.theme", theme);
    } catch {
        /* Optional preference. */
    }
};

render();
setWeb(false);
refreshStatus();
setInterval(refreshStatus, 1800);
window.addEventListener("beforeunload", () => {
    if (currentRequest) {
        fetch("/api/cancel", {
            method: "POST",
            keepalive: true,
            headers: {
                "Content-Type": "application/json",
                "X-Local-Chat-Token": token,
            },
            body: JSON.stringify({ id: currentRequest }),
        }).catch(() => {});
    }
});
