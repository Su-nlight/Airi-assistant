# Kokoro Wake-Word Dataset Generator

Production-grade rewrite of `kokorov1.py` — GPU-accelerated, class-based,
Docker-containerised, and fully configurable via environment variables.
Architecture mirrors the Coqui TTS generator.

---

## Directory Structure

```
kokoro-wakeword-synthesizer/
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   └── generate_dataset.py
├── output/              # Generated .wav files (volume-mounted)
└── models/              # Reserved for future model caching
```

---

## Quick Start

### GPU (recommended)

```bash
docker compose up --build
```

### CPU only

In `Dockerfile`, replace the base image:
```dockerfile
FROM python:3.11-slim
```

In `docker-compose.yml`, comment out `runtime: nvidia` and the `deploy:` block.

```bash
FORCE_CPU=true docker compose up --build
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
| `TARGET_SR` | `16000` | Output sample rate (Hz) |
| `LANG_CODE` | `a` | Kokoro language code (`a` = American English) |
| `RANDOM_SEED` | `42` | Reproducibility seed. Change between runs for variation. |
| `FORCE_CPU` | `false` | Force CPU even when CUDA is available |

After changing any variable, rebuild with `docker compose build` (cached layers keep this fast).

---

## What Changed vs `kokorov1.py`

| Concern | `kokorov1.py` | This version |
|---|---|---|
| Architecture | Flat procedural script | `Config` / `AudioProcessor` / `KokoroDatasetGenerator` classes |
| GPU support | None (CPU default) | `KPipeline(device="cuda")` — auto-detected, FORCE_CPU override |
| Config | Hardcoded constants | Environment variables |
| Batch/epoch loop | Single `while` loop | Epoch-tracked batches with per-batch stats |
| Statistics | Minimal print | Valid, rejected, attempts, success rate, avg time/sample |
| Colab dependencies | `!pip`, `files.download`, `IPython` | Removed — pure Python/Docker |
| Emotion simulation | None | 9-style amplitude + tremolo shaping (matches Coqui) |
| Output filename | Includes voice/speed/pitch | Adds emotion tag + millisecond timestamp for uniqueness |
| Deployment | Google Colab only | Docker + `docker-compose` (GPU or CPU profile) |

---

## GPU vs CPU Trade-off

Tested at 600 samples:

| Mode | Estimated Time |
|---|---|
| CUDA GPU (e.g. RTX 3060+) | ~10–20 min |
| CPU only | ~2–4 h |

---

## Voices Available

11 voices across American and British English:

```
af_heart  af_bella  af_nicole  af_sarah  af_sky
am_adam   am_michael
bf_emma   bf_isabella
bm_lewis  bm_george
```

All voices are used at random per sample for maximum acoustic diversity.