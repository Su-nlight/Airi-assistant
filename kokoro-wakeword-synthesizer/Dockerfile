# ── Base image ────────────────────────────────────────────────────────────────
# GPU build (CUDA 12.1 + cuDNN 8, Ubuntu 22.04)
# For CPU-only: swap for python:3.11-slim and remove nvidia-related layers
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# ── System dependencies ───────────────────────────────────────────────────────
# espeak-ng is required by Kokoro's phonemizer backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# Pin pip to avoid resolver surprises
RUN pip3 install --upgrade pip setuptools wheel

# ── Python dependencies ───────────────────────────────────────────────────────
# torch must be installed before kokoro so the correct CUDA-enabled wheel is used
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install \
    "kokoro>=0.1.9" \
    soundfile \
    librosa \
    numpy

# ── Working directory & volume mount-points ───────────────────────────────────
WORKDIR /app

RUN mkdir -p /app/scripts /app/output /app/models

# ── Entry-point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python3"]
CMD ["/app/scripts/generate_dataset.py"]