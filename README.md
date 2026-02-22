<div align="center">

# 🤖 AIRI Assistant

> *Your offline, privacy-first voice assistant that actually slaps.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-blue.svg)]()
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)]()
[![Edge AI](https://img.shields.io/badge/Edge-AI-green.svg)]()

**AIRI** is a lightweight, privacy-focused, edge-optimized voice assistant built for real-time wake word detection and fully on-device inference. No cloud. No subscriptions. No nonsense — just fast, local AI that runs right where you are.

</div>

---

## ✨ Features

| 🔥 What It Does | 💡 Why It's Cool |
|---|---|
| Custom wake word detection (`"Hello Airi"`) | Wakes up only for *you* |
| Fully offline wake word inference | Your voice stays on your device, period |
| INT8 quantized ML models | Blazing fast, stupidly small footprint |
| Synthetic + real voice training | Best of both worlds |
| Docker-based deployment | Spin it up in seconds on any Linux box |
| Low memory & CPU usage | Runs even on potato hardware |
| Modular architecture | Hack it, extend it, make it yours |

---

## 🏗️ System Architecture

No cap, here's how AIRI flows from your mouth to your speakers:

```
🎙️  Microphone Input
        ↓
🧠  Wake Word Detection Model (Edge Impulse)
        ↓
⚙️  Command Processing Logic
        ↓
💬  Response Generation
        ↓
🔊  Coqui TTS Synthesizer
        ↓
🔈  Audio Output
```

Clean. Simple. Hits different.

---

## 📁 Repository Structure

```
Airi-assistant/
│
├── coqui-tts-synthesizer/     # The voice — smooth and offline
├── models-versioned/          # Trained wake word models, versioned like a pro
├── Dockerfile                 # One command and you're live
├── README.md                  # You're reading it rn
├── LICENSE                    # MIT — do what you want (almost)
└── .gitignore
```

---

## 🧠 Wake Word Model Development

AIRI's brain was trained using **[Edge Impulse](https://edgeimpulse.com/)** — an edge ML platform that makes building and deploying optimized models for sensor data actually enjoyable.

### Dataset Classes

The model learned to distinguish between three classes (like a vibe check for audio):

- 🟢 **Wake Word** — `"Hello Airi"` (the magic words)
- 🟡 **Unknown Words** — random speech like "yes", "no", "left", etc.
- 🔴 **Noise** — background sounds, silence, environmental chaos

> At least **10 minutes** of wake word audio was collected to make sure AIRI actually knows when you're talking to it and not just vibing in the background.

### Dataset Variants

**Naive Dataset** — real human recordings with high variance in accents, pitch, tone, and speaking style, plus keyword spotting datasets from Edge Impulse. Real talk: diversity in data = robustness in the wild.

**Synthetic Dataset** — generated using the **Coqui TTS synthesizer** (`coqui-tts-synthesizer/`). Why record thousands of samples manually when you can just... generate them? Big brain move. Benefits:
- Scales like crazy
- Consistent pronunciation
- Way less manual effort

---

## 🎛️ Signal Processing & Feature Extraction

AIRI uses **MFCC (Mel-Frequency Cepstral Coefficients)** to turn raw audio into something the model can actually learn from. Think of it as translating sound into math that hits different.

MFCCs help by:
- Converting raw audio into meaningful features
- Cutting out the noise (literally)
- Highlighting the frequency patterns that matter for speech

Feature Explorer visualization was used to validate dataset quality and class separation — because shipping blind is not the move.

---

## 🏋️ Model Training

Training was done end-to-end on the **Edge Impulse** platform, which handles everything from data ingestion to deployment optimization. Here's what went down:

- **Feature extraction** via the MFCC signal processing block
- **Neural network training** using Edge Impulse's built-in classifier — it learned the wake word patterns so you don't have to hardcode anything
- **Confusion matrix analysis** to catch misclassifications before they ship
- **Performance validation** on unseen test data — no shortcuts, no cap
- **INT8 quantization** for the final model — tiny, fast, and ready for edge devices

### Train/Test Split

```
80% Training  ████████████████░░░░  
20% Testing   ████░░░░░░░░░░░░░░░░  
```

This setup ensures low false positives and reliable real-world detection. AIRI won't wake up every time someone sneezes near your mic.

---

## 🚀 Deployment

AIRI currently runs on **Linux via Docker**. Setup is genuinely easy — no PhD required.

### Requirements

- Linux-based OS
- Docker installed
- Microphone access
- Audio output device

### Let's Get It Running

```bash
# Build the container
docker build -t airi-assistant .

# Run it (with microphone access — mandatory, not optional lol)
docker run --device /dev/snd airi-assistant
```

That's it. You're up.

---

## 🗣️ Text-to-Speech Engine

AIRI uses **[Coqui TTS](https://github.com/coqui-ai/TTS)** for speech synthesis — fully offline, sounds natural, and responds fast. No weird robotic voices, no cloud dependency.

**Directory:** `coqui-tts-synthesizer/`

---

## 🎯 Use Cases

Whether you're a hobbyist or an edge AI nerd, AIRI's got you:

- 🏠 **Smart home integration** — control your space with your voice
- 🧪 **Edge AI experimentation** — perfect playground for ML tinkering
- 🔒 **Offline voice control** — zero data leaving your device
- 💻 **Personal voice assistant** — your own JARVIS (kinda)
- 🔧 **Embedded AI projects** — slap it on your Raspberry Pi and go wild

---

## 🔮 What's Coming Next

The roadmap is looking bussin':

- [ ] Full command recognition pipeline
- [ ] LLM integration for actual conversations (not just wake words)
- [ ] Cross-platform native support (Windows & macOS gang, you're not forgotten)
- [ ] Improved voice quality
- [ ] Hardware device support

Stay tuned. AIRI is just getting started. 🚀

---

## 📄 License

MIT License — free to use, modify, and distribute. Just don't be weird about it.

---

<div align="center">

**Made with 💙 for edge AI enthusiasts who believe your data should stay yours.**

*Say "Hello Airi" and let's get to work.*

</div>
