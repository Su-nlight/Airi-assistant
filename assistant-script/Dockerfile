FROM python:3.9-slim

# Install system dependencies (include libs needed by OpenCV headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      portaudio19-dev \
      libportaudio2 \
      libglib2.0-0 \
      libgl1 \
      pulseaudio \
      alsa-utils \
      alsa-plugin-pulse \
      wget \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages in one pip command (headless OpenCV avoids many GUI deps)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.24.3 \
      six \
      pyaudio \
      "opencv-python-headless>=4.5.1.48,<5" \
      edge-impulse-linux

# Verify installations (also check cv2 import)
RUN python - <<'PY'
import importlib, sys
modules = ("edge_impulse_linux","pyaudio","numpy","cv2")
for m in modules:
    try:
        importlib.import_module(m)
        print("✓", m)
    except Exception as e:
        print("✗", m, e)
        sys.exit(1)
PY

RUN mkdir -p /app/logs

COPY keyword_spotter.py /app/keyword_spotter.py

CMD ["python", "-u", "keyword_spotter.py"]
