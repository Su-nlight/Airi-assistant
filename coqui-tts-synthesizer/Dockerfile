# FROM ghcr.io/coqui-ai/tts-cpu:latest        
# for cpu only processing

FROM ghcr.io/coqui-ai/tts:latest
# for GPU enabled processing

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TTS_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install TTS and dependencies
RUN pip3 install TTS scipy soundfile

# Create working directory
WORKDIR /app

# Create directories for models and output
RUN mkdir -p /app/models /app/output /app/scripts

# Set the entrypoint
ENTRYPOINT ["python3"]