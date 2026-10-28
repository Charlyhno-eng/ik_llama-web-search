# Local Chat

Chat with local **GGUF models** in a simple web interface, with contextual **DuckDuckGo search** and clickable sources.

A fork of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), itself derived from [llama.cpp](https://github.com/ggml-org/llama.cpp). Our additions include the local chat application and CPU optimizations for tied ternary output weights; see [Local Chat](docs/local-chat.md).

## Install

Requires **Python 3.10+**, **CMake** and a **C++20 compiler**. Install the prerequisites for your system once; no Python packages or Node.js are needed.

### Windows (PowerShell)

With [WinGet](https://learn.microsoft.com/en-us/windows/package-manager/winget/):

```powershell
winget install --exact --id Git.Git
winget install --exact --id Python.Python.3.13
winget install --exact --id Kitware.CMake
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Wait for installation to finish, approve administrator prompts if requested, and reopen PowerShell. Restart Windows first if the installer requires it. The [C++ build tools](https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2022) include the compiler and Windows SDK.

### Linux (Ubuntu/Debian)

```sh
sudo apt update
sudo apt install git python3 cmake build-essential
```

On other distributions, install equivalent packages. Check that `python3 --version` reports 3.10 or newer.

### macOS (Terminal)

Install Apple's command line tools, then wait for their installation dialog to complete:

```sh
xcode-select --install
```

If the tools are already installed, skip that step. With [Homebrew](https://brew.sh/) installed:

```sh
brew install python@3.13 cmake
```

### Download the project

On any platform (skip this step if you already have the project):

```sh
git clone https://github.com/Charlyhno-eng/ik_llama-web-search.git
cd ik_llama-web-search
```

## Start

From the project directory, on **Windows, Linux or macOS**:

```sh
python start.py
```

If your Python command is `python3`, use `python3 start.py`; on Windows, you can also use `py -3 start.py`.

Your browser opens at **http://127.0.0.1:8080**. Open **Settings**, choose a `.gguf` file, and select **Save & load**. Models are supplied separately. Your model and settings are remembered for the next launch. Press **Ctrl+C** in the terminal to stop the application and release the model.

If no compiled `llama-server` is present, the application builds it on first model load; this can take several minutes. An existing native server binary avoids the CMake/compiler requirements. Linux/macOS binaries are named `llama-server`; Windows binaries are named `llama-server.exe`.

## Features

- Local GGUF file browser, remembered model path, **2,048–8,192 token context** and adjustable CPU threads.
- Native **LFM3-1.7B GGUF** support alongside Qwen and Bonsai, using the same optimized engine. See [validation and performance](docs/performance-lfm3-2026-10-28.md); existing checkouts with an older binary must rebuild `llama-server` as described in [Local Chat](docs/local-chat.md#lfm3-text-models).
- Streaming chat, stop button, saved conversations, navy/light themes and mobile layout. The interface is English; the assistant follows your message's language.
- **Web search** understands follow-up questions, reformulates ambiguous queries locally, cross-checks focused searches, filters unrelated results and reads relevant public pages. URLs supplied in your message are opened directly; earlier sources can be revisited for fresh evidence.
- Up to **4 sources**, with page/search excerpt labels and links built only from retrieved URLs. Evidence is capped at **600 words**, then fitted to the actual model's token budget with instructions, history and space for the answer.
- Response **tokens/second** and **time to first token**. First-token time includes search and preparation; hover for the model-only delay. Speed uses the engine's timing when available, with a labeled estimate otherwise.

Inference stays local. Web mode sends contextual queries, which can include the subject of earlier messages, to DuckDuckGo and requests relevant public pages. Search results are cached in memory for three minutes; page content is read again. DuckDuckGo can block automated access, and some pages cannot be read. The app reports missing evidence and can still use directly readable sources; neither search coverage nor model accuracy is guaranteed.

Settings defaults: **4,096 context tokens**, **1,024 maximum response tokens**, **0.5 temperature** and **4 CPU threads**. All are adjustable in Settings. A separate prompt CPU thread setting accepts 0–256; zero follows the generation thread count. Response length accepts 128–4,096 tokens and is capped at half the context; reserving more response tokens leaves less room for history and web sources. Temperature accepts 0–2: lower values are more predictable, higher values more varied; this does not guarantee accuracy.

Settings and logs live in `.local-chat/` (ignored by Git). Changing settings in the web interface never changes the shared defaults or tracked files, so ordinary commits and pushes exclude your local preferences and model path. Conversations stay in your browser's local storage. Existing conversations and theme preferences are preserved across this update.

Other port: `python start.py --port 8081`. No automatic browser: `python start.py --no-browser`. Set the optional `LLAMA_SERVER` environment variable to use another native server executable.

[Technical details and tests](docs/local-chat.md) · [MIT license](LICENSE)
