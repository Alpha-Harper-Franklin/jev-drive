# First recorded-image diagnostic — 2026-09-21

## Scope and provenance

This run tests a multimodal observation-to-decision pipeline. It is not a driving benchmark and has no accuracy or safety labels. The source is private recorded BeamNG RGB: 740 locally available frames across 9 source groups. The first, middle, and last available frame in each group yield 27 images. This is a development convenience sample, not an independent-scene test set. Images, private paths, and original project code are not distributed.

All input images, five model-weight shards, model configuration, tokenizer, manifest, and experiment script have SHA-256 records in private run metadata. A completion receipt matches all 27 outputs. The GPU process completed and exited naturally.

## Configuration

- Qwen2.5-VL-7B-Instruct, matching model class, bfloat16, SDPA, greedy inference.
- One RTX 4090, PyTorch 2.13.0+cu130, Transformers 5.15.0, Python 3.10.12.
- Maximum image budget: 401,408 pixels; caption cap: 160 new tokens.
- Warm-up excluded; image reading and preprocessing included as described in the summary tool.
- One generic task: follow the visible route cautiously; destination and motion history unavailable.
- Four supervisory candidates and two orderings, with no action execution.
- Qwen decisions use conditional next-token scores over A/B/C/D. They are uncalibrated; median total candidate probability mass exceeded 0.999 in both baselines.
- Returned Jev model: `jev-1.13.0`. Calls originated from the Windows client, so timing includes network overhead. No automatic retries.

## Results

| Path | Valid / attempted | Decision p50 / p95 | Per-image pipeline p50 / p95 |
| --- | ---: | ---: | ---: |
| Image → Qwen choice | 54 / 54 | 267 / 293 ms | 275 / 302 ms |
| Image → caption → Qwen choice | 54 / 54 | 94 / 110 ms | 2,848 / 4,057 ms |
| Image → caption → Jev | 47 / 54 | 774 / 2,067 ms | 3,966 / 5,468 ms |

Pipeline timings include one description generation for each caption path and exclude model loading. Caption latency was p50 2,728 ms and p95 3,895 ms. No caption reached the token cap. The prompted 80-word preference is not a hard word-count constraint.

Jev success-only percentiles exclude seven failed attempts. Across all 54 attempts, request latency was p50 893 ms and p95 2,366 ms. One transport/JSON error occurred at approximately 15.4 seconds, consistent with the socket timeout. Five failures were HTTP 503. One response was rejected by the client validator; its raw payload was not retained, so the precise cause remains unresolved. A separate diagnostic retry also received HTTP 503 and does not replace any original attempt.

Jev reported 27,488 input tokens for validated responses. At the published $0.042 per million input tokens, these known responses imply approximately $0.001155. Total billing remains unknown because failed and rejected attempts have incomplete usage. GPU cost is excluded.

Reordering changed 0/27 image-Qwen choices, 3/27 caption-Qwen choices, and 0/23 fully observed Jev pairs. For the first ordering, Jev disagreed with caption-Qwen on 16/24 shared valid samples and image-Qwen on 21/24. These are different requests, not mistakes or successes: no ground truth adjudicates them. No validated response selected `replan` or `defer`, so this small run does not exercise the full recovery vocabulary.

## Consequences

The naive per-frame caption-to-Jev route is slower than local one-token visual scoring here. The direct visual baseline also requests more observations, while Jev usually requests continuation; latency alone cannot decide which behavior is better. The next meaningful test needs executed recovery outcomes, balanced disturbances, temporal state, and a strong local-classifier baseline.

Occasional semantic requests over reusable observations, with explicit stale-response rejection, are a hypothesis for further testing. More unlabelled adjacent frames would not resolve the missing outcome evidence.

Aggregate results: [probe-summary.json](results/probe-summary.json). Commands: [experiments/README.md](../experiments/README.md). Private raw records remain outside this repository. Users can run the scripts on licensed images, but cannot reproduce these exact values from the public repository alone.
