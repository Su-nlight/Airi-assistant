# Piper TTS Wake-Word Dataset Generator

Generates a large, acoustically diverse set of synthetic wake-word samples
("Hello Airi") using [Piper TTS](https://github.com/rhasspy/piper) — a fast,
fully local neural TTS engine backed by ONNX/VITS models.

Architecture mirrors the Coqui TTS and Kokoro generators in this repo.

---

## Directory Structure

```
piper-wakeword-synthesizer/
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   └── generate_dataset.py
├── output/              # Generated .wav files (volume-mounted)
└── models/              # Downloaded Piper .onnx models (volume-mounted)
```

---

## Quick Start

### GPU (recommended)

```bash
docker compose up --build
```

### CPU only

In `Dockerfile`, swap the base image:

```dockerfile
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM python:3.11-slim
```

Also replace `onnxruntime-gpu` with `onnxruntime` in the `pip3 install` step.

In `docker-compose.yml`, comment out `runtime: nvidia` and the `deploy:` block:

```yaml
    # runtime: nvidia
    environment:
      ...
      - FORCE_CPU=true
    # deploy:
    #   resources: ...
```

Then rebuild:

```bash
docker compose build
docker compose up
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WAKE_WORD` | `Hello Airi` | Text to synthesise |
| `TARGET_SAMPLES` | `600` | Number of accepted `.wav` files to produce |
| `MIN_DURATION` | `1.0` | Minimum accepted clip length (seconds) |
| `MAX_DURATION` | `1.3` | Maximum accepted clip length (seconds) |
| `BATCH_SIZE` | `20` | Samples attempted per epoch |
| `OUTPUT_DIR` | `/app/output` | Output path inside the container |
| `MODELS_DIR` | `/app/models` | Model download path inside the container |
| `TARGET_SR` | `22050` | Output sample rate (Hz). Use `16000` for downstream ML pipelines |
| `RANDOM_SEED` | `42` | Reproducibility seed. Change between runs for variation |
| `FORCE_CPU` | `false` | Force CPU even when CUDA is available |

After changing any variable, rebuild with `docker compose build` (cached layers keep this fast).

---

## Voices

10 pre-configured English voices across male/female and US/GB accents:

| Voice key | Gender | Accent | Quality |
|---|---|---|---|
| `en_US-lessac-medium` | Female | American | Medium |
| `en_US-libritts-high` | Mixed | American | High |
| `en_US-amy-medium` | Female | American | Medium |
| `en_US-hfc_female-medium` | Female | American | Medium |
| `en_US-ryan-medium` | Male | American | Medium |
| `en_US-hfc_male-medium` | Male | American | Medium |
| `en_US-joe-medium` | Male | American | Medium |
| `en_GB-alan-medium` | Male | British | Medium |
| `en_GB-jenny_dioco-medium` | Female | British | Medium |
| `en_GB-vctk-medium` | Mixed | British | Medium |

Models are downloaded automatically from the Piper voice registry on first run.

---

## Variation Pipeline

Each sample is randomised across five independent axes:

| Axis | Values |
|---|---|
| **Voice** | 10 voices (male/female, US/GB) |
| **Speed** (`length_scale`) | 0.8 → 1.2 (5 levels) |
| **Pitch shift** | −3 → +3 semitones (7 levels) |
| **Audio noise** (`noise_scale`) | 0.333 → 0.800 (4 levels) |
| **Emotion** | 9 styles via amplitude + tremolo shaping |

This gives a theoretical combination space of **10 × 5 × 7 × 4 × 9 = 12,600** unique
configurations, far exceeding the 600-sample default target.

### Piper-Specific Notes

- **`length_scale`** — Piper's native speed control; >1 = slower, <1 = faster.
  Preferred over post-synthesis time-stretch for better naturalness.
- **`noise_scale`** — Controls stochastic audio generation (variation in
  pronunciation quality). Higher = more variation.
- **`noise_w_scale`** — Controls duration stochasticity (variation in timing
  between phones). Higher = more timing variation.
- **Duration gate** — Samples outside `[MIN_DURATION, MAX_DURATION]` are
  time-stretched to the window midpoint via `librosa.effects.time_stretch`.
  Only pathological cases (ratio >2× or <0.5×) are hard-rejected.

---

## Why Piper?

| Concern | Coqui TTS | Kokoro | **Piper TTS** |
|---|---|---|---|
| CPU speed | ⭐⭐ Slow | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Very fast |
| GPU speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory footprint | High | Moderate | Low |
| Install complexity | Medium | Medium | Easy (pip only) |
| Voice diversity | 109+ speakers | 11 voices | 50+ voices, 30+ languages |
| Piper is designed to run well even on CPU-only hardware (Raspberry Pi 4). | | | |

### GPU vs CPU Trade-off

Because Piper is ONNX-based and highly optimised, even CPU inference is fast
(typically 0.05–0.3 s per sample). Realistic estimates for 600 samples:

| Mode | Estimated Time |
|---|---|
| CUDA GPU (e.g. RTX 3060+) | ~5–10 min |
| CPU only | ~15–45 min |

---

## Conclusion

Piper TTS is the fastest and lightest of the three approaches in this repo.
It provides excellent voice diversity, runs offline, and requires no API keys.
The primary trade-off is that, unlike Kokoro or Coqui, Piper does not expose
fine-grained multi-speaker control within a single model — voice diversity is
achieved by loading multiple model files instead.