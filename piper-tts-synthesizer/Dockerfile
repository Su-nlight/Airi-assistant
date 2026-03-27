# ── Base image ────────────────────────────────────────────────────────────────
# Use the official pre-built Wyoming Piper image as base.
# This already contains piper, espeak-ng, Python and all system dependencies
# so zero apt-get calls are needed — no network issues during build.
FROM rhasspy/wyoming-piper:latest

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Extra Python dependencies only (pip uses HTTPS port 443) ─────────────────
RUN pip3 install --upgrade pip && \
    pip3 install \
        onnxruntime \
        numpy \
        soundfile \
        librosa \
        requests

# ── Working directory & volume mount-points ───────────────────────────────────
WORKDIR /app

RUN mkdir -p /app/scripts /app/output /app/models

# ── Entry-point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python3"]
CMD ["/app/scripts/generate_dataset.py"]