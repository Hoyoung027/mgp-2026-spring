# HW6 Triton ResNet18 Experiment Log

## Environment

- Server: `server8`
- Working directory: `~/mgp-2026-spring/hw6/experiment`
- Command: `make triton`
- Model: Triton ResNet18
- Accuracy target: at least 75%
- Time target: within 200 ms

## Current Conv2d Configuration

- Single generic Conv2d Triton kernel
- No separate `1x1` Conv2d fast path
- Tile parameters:
  - `BLOCK_M = 32`
  - `BLOCK_N = 32`
  - `BLOCK_K = 64`

## Results

The Triton cache was cleared before the first run, so the first execution includes Triton JIT compilation overhead.

| Run | Inference Time (ms) | Accuracy (%) | Note |
|---:|---:|---:|---|
| 1 | 239.2161 | 79.83 | Cold run, includes JIT/cache generation |
| 2 | 47.6074 | 79.83 | Cached run |
| 3 | 47.5425 | 79.83 | Cached run |
| 4 | 47.7113 | 79.83 | Cached run |
| 5 | 47.8845 | 79.83 | Cached run |
| 6 | 47.8091 | 79.83 | Cached run |

## Summary

- Accuracy is stable at `79.83%`, so the implementation satisfies the correctness requirement.
- Cached inference time is stable around `47.5-47.9 ms`.
- The cold run takes `239.2161 ms`, which exceeds the 200 ms target if the grading includes Triton JIT compilation time.
- Compared with the earlier `BLOCK_K = 32` configuration, this `BLOCK_K = 64` configuration is slower in both cold and cached runs.

## Interpretation

Increasing `BLOCK_K` from 32 to 64 reduces the number of reduction-loop iterations, but it also increases the per-program tile size and compile/runtime cost. In this experiment, that tradeoff was unfavorable:

- Cold-start time increased significantly due to heavier JIT compilation.
- Cached runtime also increased from about `43 ms` to about `48 ms`.

Therefore, the safer submission candidate is the previous single-kernel Conv2d configuration with `BLOCK_K = 32`.
