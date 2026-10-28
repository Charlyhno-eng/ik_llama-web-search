# LFM3 native support and CPU validation — October 28, 2026

The original load failure was `unknown model architecture: 'lfm3'`, rather than
an out-of-memory error. Native support was added to the existing ik_llama engine,
following [the official implementation](https://github.com/ggml-org/llama.cpp/pull/31252)
at commit `19cce6919992fdffd92e0ac152caaf1e2b8f254e` (merged October 27).
No alternate runtime is selected for existing models.

The implementation loads LFM3's output norm, sandwich norms, per-head attention
gates and shared global RoPE factors. It uses the existing short-convolution
state path and gated GQA kernels, with distinct masks and RoPE bases for local
and global attention. The GGUF is never modified. Qwen and Bonsai graphs and
quantization kernels are unchanged. In this build, the before/after `libggml.so`
files have the **same SHA-256**:
`4f82d8f300797f14c64f76487fc14cb0be68144c37081c01f4e00df391ecaa58`.

## Method

Intel Core i7-1195G7 (4 cores / 8 threads), 16 GiB RAM, CPU inference, four
generation and prompt threads, 4,096-token context, Release / `GGML_NATIVE=ON`.
Only one benchmark server generates at a time. Before binaries and their shared
libraries were copied before building the change, and selected with a separate
`LD_LIBRARY_PATH` during the comparison.

`scripts/benchmark-local.py` sends the same French prompt token IDs to both
variants, disables prompt reuse, and requests exactly 128 generated tokens with
greedy sampling, seed 42 and EOS ignored. Each case has one excluded warmup and
three measured requests; results are medians from engine timings, not SSE chunk
counts. The runner checks both actual prompt and generated token counts. LFM3 is
also tested with 512- and 2,048-token prompts. Existing models use 64-token prompts.

These are local, serial comparisons, subject to clock/temperature and background
load variation. Historical headline speeds are useful context, but the preserved
before binary is the control for this change. There is no new measurement of
classic llama.cpp for the Qwen/Bonsai rows; their previously reported percentage
gains are not recalculated from unrelated runs.

## LFM3 performance

| Prompt tokens | ik TG (tok/s) | Official TG (tok/s) | TG change | ik PP (tok/s) | Official PP (tok/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 25.28 | 24.20 | +4.49% | 195.85 | 161.46 |
| 512 | 25.93 | 23.63 | +9.75% | 337.91 | 178.65 |
| 2048 | 25.38 | 22.23 | +14.18% | 326.17 | 175.01 |

TG is generation throughput; PP is prompt processing. LFM3 remains around
25–26 generated tokens/s across these prompt sizes. Different architectures,
quantizations and parameter counts do not imply equal absolute speeds.

## Existing-model regression comparison

All speeds below are generated tokens/s. Historical values are the user-provided
reference; before/after values are this controlled local campaign.

| Model | Quantization | Historical | Before | With LFM3 support | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B | Q4_K_M | 52.6 | 54.442 | 55.838 | +2.56% |
| Qwen 3.5 2B | Q5_K_M | 18.4 | 18.819 | 18.897 | +0.42% |
| Qwen 3.5 4B | Q4_K_M | 10.3 | 10.075 | 10.084 | +0.09% |
| Qwen 3.5 9B | IQ4_XS | 6.3 | 6.040 | 6.058 | +0.30% |
| Bonsai 2 2B | PQ2_0 | 46.6 | 47.251 | 47.203 | -0.10% |
| Bonsai 2 4B | PQ2_0 | 21.0 | 22.023 | 22.150 | +0.58% |
| Bonsai 2 9B | PQ2_0 | 12.1 | 12.729 | 12.758 | +0.23% |
| Bonsai 2 27B | PQ2_0 | 3.9 | 3.958 | 3.984 | +0.68% |

All eight models completed the requested generations. For every model, the
before/after generation hashes and prompt hashes match. The small throughput
differences in this campaign do not establish a new optimization; no material
performance regression was observed under these conditions. Other contexts,
hardware and background loads are not covered by this measurement.

## Functional and numerical checks

`scripts/verify-lfm3.py` uses the actual app loader and streaming chat with an
isolated temporary settings directory. LFM3 answered “Paris.”, then “La France.”
to a follow-up. Cancelling after the first text chunk and starting a new chat
also passed. Real engine timing fields were received.

Next-token probabilities were compared after 64, 255, 256, 257, 512 and 1,024
prompt tokens, including the sliding-window boundary. Resetting the convolution
state after an unrelated prompt reproduced the probabilities exactly. Prefill
with 32-token microbatches differed by at most 0.001111, and comparison with
the official CPU build by at most 0.000565 (absolute probability difference;
top-64 lists, with missing entries treated as zero). The checker enforces a
0.002 tolerance for chunking/reference comparisons. This checks representative
numerical behavior, not a general model-quality benchmark.

The 66 existing Python tests pass, JavaScript syntax validation passes, and
`test-cpu-buffer` / `test-tied-output` pass with the native engine.

## Reproduction and artifacts

Local configs, raw per-request timings, generation/prompt hashes, full server
commands and logs are in `.local-chat/lfm3-validation/` (Git-ignored).

```sh
python scripts/verify-lfm3.py /path/to/LFM3-1.7B-Q4_K_M.gguf .local-chat/lfm3-check.json \
  --reference /path/to/llama.cpp-19cce69/llama-server
python scripts/benchmark-local.py .local-chat/lfm3-validation/lfm3-config.json .local-chat/new-lfm3.jsonl
python scripts/benchmark-local.py .local-chat/lfm3-validation/regression-config.json .local-chat/new-regression.jsonl
```

The benchmark refuses to overwrite an existing results file. For an official
llama.cpp variant, set `"runtime": "llama.cpp"` to omit ik-specific flags.
The reference build is used only for validation; the application requires only
its bundled engine and the existing GGUF.
