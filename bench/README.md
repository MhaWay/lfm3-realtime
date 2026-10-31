# Benchmarks

Reproduce with `python bench/latency.py --in sample.wav` while `lfm3-realtime` runs.

## 2026-10-31 — DGX Spark (GB10), bf16, interleaved mode

Input: 13 s English speech sample; one response per run.

| Test | First audio | Wall clock | Output audio |
|---|---|---|---|
| response.create → first `response.audio.delta` | 1.28 s | 2.7-3.4 s | 3.8-5.5 s |
| sequential mode, first audio frame | 0.79 s | — | — |

Notes:
- Model: `LiquidAI/LFM3-Audio-2B`, local snapshot.
- First run includes page-cache warmup; steady-state deltas land within the same budget.
