# Local Chat

Chat with local **GGUF models** in a simple web interface, with contextual **DuckDuckGo search** and clickable sources.

A fork of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), itself derived from [llama.cpp](https://github.com/ggml-org/llama.cpp). Our addition is the local chat application; **the inference engine is unchanged**.

## Start

From the project directory, on **Windows, Linux or macOS**:

```sh
python start.py
```

Your browser opens at **http://127.0.0.1:8080**. Open **Settings**, choose a `.gguf` file, and select **Save & load**. Your model and settings are remembered for the next launch. Press **Ctrl+C** in the terminal to stop the application and release the model.

**Requirements:** Python 3.10+ available in your terminal. If your system names it `python3`, use `python3 start.py` (or `py -3 start.py` with the Windows Python launcher). No Bash, Python packages or Node.js are needed.

If no compiled `llama-server` is present, the application builds it on first model load; this can take several minutes. Install **CMake** and a **C++20 compiler** first:

- **Windows:** Visual Studio 2022 Build Tools with **Desktop development with C++**, including the Windows SDK and CMake tools. Reopen your terminal after installation.
- **Linux (Ubuntu/Debian):** `sudo apt install python3 cmake build-essential`.
- **macOS:** Xcode Command Line Tools (`xcode-select --install`) and CMake.

An existing native server binary avoids the build requirements. Linux/macOS binaries are named `llama-server`; Windows binaries are named `llama-server.exe`.

## Features

- Local GGUF file browser, remembered model path, **2,048–8,192 token context** and adjustable CPU threads.
- Streaming chat, stop button, saved conversations, navy/light themes and mobile layout. The interface is English; the assistant follows your message's language.
- **Web search** understands follow-up questions, reformulates ambiguous queries locally, cross-checks focused searches, filters unrelated results and reads relevant public pages. URLs supplied in your message are opened directly; earlier sources can be revisited for fresh evidence.
- Up to **5 sources**, with page/search excerpt labels and links built only from retrieved URLs. Evidence is capped at **750 words**, then fitted to the actual model's token budget with instructions, history and space for the answer.
- Response **tokens/second** and **time to first token**. First-token time includes search and preparation; hover for the model-only delay. Speed uses the engine's timing when available, with a labeled estimate otherwise.

Inference stays local. Web mode sends contextual queries, which can include the subject of earlier messages, to DuckDuckGo and requests relevant public pages. Search results are cached in memory for three minutes; page content is read again. DuckDuckGo can block automated access, and some pages cannot be read. The app reports missing evidence and can still use directly readable sources; neither search coverage nor model accuracy is guaranteed.

Settings and logs live in `.local-chat/` (ignored by Git). Conversations stay in your browser's local storage. Existing conversations and theme preferences are preserved across this update.

Other port: `python start.py --port 8081`. No automatic browser: `python start.py --no-browser`. Set the optional `LLAMA_SERVER` environment variable to use another native server executable. The previous `start.sh` remains a compatibility shortcut on Linux/macOS.

[Technical details and tests](docs/local-chat.md) · [Upstream documentation](README.upstream.md) · [MIT license](LICENSE)
