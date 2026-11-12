# Raon support: CPU performance and regression check — November 12, 2026

Raon-1 nano Q4_K_M generates about **21 tokens/s** on this machine. No material
generation regression was observed on the nine existing Qwen, Bonsai and LFM3
models. This describes the tested prompts and settings, not every workload.

## Method and scope

Intel Core i7-1195G7, 16 GiB RAM, Linux, CPU inference, four generation and
prompt threads, 4,096-token context, runtime repacking enabled. Requests use
identical token IDs, greedy decoding, seed 42, prompt reuse disabled and exactly
128 generated tokens (EOS ignored). The synthetic French prompt is the existing
`scripts/benchmark-local.py` workload. Each case has one excluded warmup and
three measured requests. Rates are medians of native engine timings.

The comparison loads the actual previous Python Engine class from Git commit
`0b550c9a84077d2e59576744ce6806a9f523d412` and the current class, each using disposable settings. Only one
model/server runs at a time; variant order alternates between models. This
avoids retaining two copies of large models in RAM. These are serial blocks,
so clock, temperature and background activity can affect small differences.

For every non-Raon model, the complete server commands match after normalizing
the ephemeral port, including the same absolute binary path. Generation hashes
also match for every before/after pair. There are no changes in `src/`, `ggml/`
or `common/` relative to the baseline. The Raon integration only selects a
separate engine for the `raon1nano` architecture; it does not rebuild or replace
the engine or shared libraries used by the other models.

## Raon-1 nano Q4_K_M

| Input tokens | Generation (tok/s) | Prompt processing (tok/s) |
| ---: | ---: | ---: |
| 64 | 21.24 | 182.91 |
| 512 | 20.99 | 282.15 |
| 2048 | 20.71 | 240.68 |

Prompt processing precedes generation. For example, 2,048 uncached input tokens
took a median 8.51 seconds to process; the approximately 21 tok/s generation
figure does not include that initial wait or web retrieval.

## Existing models

Inputs are 64 tokens; outputs are 128 tokens. All rates below are generated
tokens/s. Quantizations follow the local GGUF filenames.

| Model | Before | After | Change |
| --- | ---: | ---: | ---: |
| LFM3-1.7B-Q4_K_M | 26.093 | 26.022 | -0.27% |
| Qwen3.5-0.8B-M-TS-Q4_K_M | 55.118 | 55.615 | +0.90% |
| Qwen3.5-2B-Q5_K_M | 18.821 | 18.842 | +0.11% |
| Qwen3.5-4B-Q4_K_M | 10.023 | 10.066 | +0.42% |
| Qwen3.5-9B-IQ4_XS | 6.034 | 6.020 | -0.23% |
| Ternary-Bonsai-2-27B-PQ2_0 | 3.945 | 3.976 | +0.78% |
| Ternary-Bonsai-2-2B-PQ2_0 | 48.371 | 47.131 | -2.56% |
| Ternary-Bonsai-2-4B-PQ2_0 | 22.221 | 22.165 | -0.25% |
| Ternary-Bonsai-2-9B-PQ2_0 | 12.799 | 12.779 | -0.16% |

The initial Bonsai 2B comparison had a larger difference, but its baseline
samples ranged from 46.46 to 49.89 tok/s. A follow-up kept both small-model
servers loaded and alternated requests in AB/BA order, with seven measured
requests per version plus excluded warmups, generating on only one server at
a time. Its medians were **47.432 → 47.247 tok/s (-0.39%)**;
output hashes match. The initial −2.56% difference was not reproduced by that
control. This supports normal measurement variation rather than a persistent
slowdown; these measurements are not a formal statistical equivalence test.

## Reproduction and artifacts

```sh
python3 scripts/benchmark-raon-regression.py /path/to/IA-models .local-chat/raon-performance/new-run \
  --baseline-ref 0b550c9a84077d2e59576744ce6806a9f523d412
```

Build the Raon backend once by loading the model before running the benchmark.
The runner leaves saved app settings unchanged. Stop other inference sessions
while benchmarking and restart `python3 start.py` afterward.

Local artifacts (Git-ignored):

- `.local-chat/raon-performance/run1/measurements.jsonl`: all timings, prompts/output hashes and commands.
- `.local-chat/raon-performance/run1/summary.json`: medians, ranges and equality checks.
- `.local-chat/raon-performance/run1/metadata.json`: source/binary hashes and loader commands.
- `.local-chat/raon-performance/run1/system.json`: CPU, memory and platform.
- `.local-chat/raon-performance/run1/memory.jsonl`: memory and major-page-fault samples during later cases.
- `.local-chat/raon-performance/bonsai2b-abba/`: alternating-control timings and summary.

Native executable and library SHA-256 values used for the nine existing models:

```text
b7c0e3cbdd71ba26a58adce1308a4fb07f72b12e045bb0676b6ef0a1d2a803e3  build/bin/llama-server
c0a5908e2c34e887a8f4bd82217f1a2143cc03b03c964b81b5f8773602fa6d6a  build/src/libllama.so
4f82d8f300797f14c64f76487fc14cb0be68144c37081c01f4e00df391ecaa58  build/ggml/src/libggml.so
```
