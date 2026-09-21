# Recorded-image probe

This is an inference diagnostic, not a driving benchmark. It uses actual RGB images supplied by the operator; no dataset is included in this repository.

Create a JSONL manifest with one object per image:

```json
{"sample_id":"route_a_001","group_id":"route_a","image_path":"/data/images/001.png","image_sha256":"REPLACE_WITH_SHA256","provenance":"recorded_simulator_RGB","task":"Follow the visible drivable route cautiously. The destination and temporal state are unavailable."}
```

Use a locally downloaded Qwen2.5-VL-7B-Instruct model and a CUDA GPU. The code loads its matching `Qwen2_5_VLForConditionalGeneration` class with no mismatched-weight override. Model files and images are hashed before inference.

```bash
python -m pip install -e '.[vision]'
CUDA_VISIBLE_DEVICES=0 python experiments/vision_probe.py \
  --model /data/models/Qwen2.5-VL-7B-Instruct \
  --manifest /data/manifest.jsonl --output runs/vision-probe
```

The probe generates one visual description per image, then evaluates image-input and shared-caption decisions using two candidate orderings. Letter-token probabilities are renormalized over the four candidates and are **uncalibrated**; the total probability mass assigned to those letters is also logged. Reordering diagnostics are paired measurements, not extra independent samples.

Provide `TYPESAFE_API_KEY` securely in the environment and replay the descriptions:

```bash
python -m jev_drive replay --input runs/vision-probe/vision.jsonl \
  --output runs/vision-probe/jev.jsonl --orders 2
python experiments/summarize_probe.py runs/vision-probe
```

The replay sends descriptions and task text to TypeSafe. It does not send image bytes, local paths, group IDs, or outcome fields. Errors stay in the output as failed attempts; they are not replaced with rule decisions. Files are created without overwriting existing evidence.

`metadata.json`, `vision.jsonl`, `jev.jsonl`, `complete.json`, and `summary.json` provide provenance and timing. Caption generation is charged to the caption pipelines when computing per-image latency. Model loading and warm-up are recorded separately. No ground-truth outcome is supplied, so this tool reports neither accuracy nor safety improvement.
