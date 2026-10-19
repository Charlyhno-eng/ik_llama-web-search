# CPU performance comparison — 2026-10-19

The retained change vectorizes the Hadamard transform with AVX-512. The first four butterfly stages run in registers; subsequent stages and normalization use vectors. It preserves the scalar operation order and normalization. It adds no weight copy, model conversion or setting change. Blocks smaller than 16 and builds without AVX-512 retain the scalar path.

The server in `build/bin/llama-server` was rebuilt with this change. The existing repacking and huge-page optimizations were already enabled in the baseline.

## Final measurements

Intel Core i7-1195G7, four physical cores/eight logical CPUs, GCC 13.3.0, Release `-O3 -march=native`, OpenMP. Both variants use four generation threads, four prompt threads, a 4,096-token context, runtime repacking and the existing flash-attention default.

These are medians of engine-reported generation tokens/s. Each request generates exactly 128 tokens, with temperature 0.5, seed 42, EOS ignored and prompt reuse disabled. Prompts use the same token IDs: a synthetic French question followed by repeated padding to reach the requested length. Search and model loading are excluded.

Both server versions stay loaded during each final comparison. Requests run **one at a time**, alternating versions and reversing order each round. One warmup per version per scenario is excluded. This reduces the effects of reloads and changing laptop temperature. The baseline executable and its libraries were copied before the first rebuild.

| Model | Prompt tokens | Runs per version | Before, tokens/s | After, tokens/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bonsai 2B PQ2 | 64 | 8 | 47.04 | 47.57 | +1.13% |
| Bonsai 2B PQ2 | 2,048 | 4 | 46.19 | 46.58 | +0.84% |
| Bonsai 4B PQ2 | 64 | 6 | 21.97 | 22.23 | +1.18% |
| Qwen3.5 2B Q5_K_M, control | 64 | 4 | 18.94 | 19.00 | +0.36%, measurement variation |

All 18 matched Bonsai pairs favored the optimized version. Qwen does not use the model-weight Hadamard transform; its paired changes range from −0.62% to +0.70% with mixed signs. Its small median difference is not attributed to the optimization.

Prompt processing improved too:

| Model | Prompt tokens | Before, prompt tokens/s | After, prompt tokens/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Bonsai 2B PQ2 | 64 | 339.76 | 364.67 | +7.33% |
| Bonsai 2B PQ2 | 2,048 | 295.04 | 311.01 | +5.42% |
| Bonsai 4B PQ2 | 64 | 128.56 | 134.07 | +4.28% |
| Qwen3.5 2B Q5_K_M, control | 64 | 184.70 | 183.11 | −0.86%, measurement variation |

The gain in generation is modest. These measurements cover the listed hardware, models, prompts and response length; they do not establish performance for every conversation or CPU. No repeatable regression was detected in these scenarios.

## Discarded experiments

The initial sequential sweeps were used for screening. Their absolute speeds vary with machine state; final claims above use the closer alternating comparisons.

| Experiment on Bonsai 2B, 64-token prompt | Generation tokens/s | Decision |
| --- | ---: | --- |
| 4 threads, reference | 46.83 | Keep existing setting |
| 2 threads | 42.08 | Slower; discard |
| 3 threads | 45.07 | Slower; discard |
| 6 threads | 43.95 | Slower; discard |
| 8 threads | 39.06 | Slower; discard |
| Pin four workers to CPUs 0–3 | 46.84 | No demonstrated gain; discard |
| `OMP_WAIT_POLICY=ACTIVE` | 46.93 | No demonstrated gain; discard |
| `OMP_WAIT_POLICY=PASSIVE` | 39.63 | Slower; discard |

Two code experiments were also reverted: combining ternary integer accumulators reduced additions without a clear inference gain; replacing the OpenMP decode barrier with the existing spin barrier did not establish a gain beyond the variation of the control run. Their source changes and associated temporary test changes were removed. Shared defaults and saved user settings remain unchanged.

## Correctness and reproduction

`test-tied-output` now compares Hadamard results **byte for byte** with a scalar reference for block sizes 2–2,048, F32/F16/Q8 inputs, multiple rows and uneven partitions across 1/4/7 workers. It also checks that inputs remain unchanged. The test passed with both the baseline and optimized libraries. The optimized `test-tied-output` and `test-cpu-buffer` CTest checks passed. Every matched generated response in the final comparisons has the same SHA-256 digest before and after.

All 66 offline Local Chat tests passed (`python3 -m unittest discover -s local_chat/tests -v`). The benchmark script was syntax-checked and exercised with real servers, including its alternating mode.

The benchmark uses only Python's standard library:

```sh
python3 scripts/benchmark-local.py config.json results.jsonl
```

The script's `--help` documents the JSON configuration. Set `interleave` to `true`, specify the two server binaries and their respective library directories, and keep the same model, thread counts, sampling parameters and lengths. It saves every timing and output digest, rejects reused/shortened prompts or incomplete generations, excludes warmups from its summary and stops owned servers on exit.

Local raw results, configurations, logs, binary snapshots and `summary.json` are preserved under `.local-chat/performance/`, which is ignored by Git. The final datasets are `interleaved-short.jsonl`, `interleaved-long.jsonl`, `interleaved-4b.jsonl` and `interleaved-qwen.jsonl`. Each has a matching `*-config.json`; rerun with a new output filename to preserve the existing measurements. Earlier exploratory datasets are retained separately.
