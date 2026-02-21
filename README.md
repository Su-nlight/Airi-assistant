# Assistant
AIRI Assistant is a lightweight, privacy-focused, edge-optimized voice assistant designed for real-time wake word detection and on-device inference. It uses custom-trained machine learning models for keyword spotting and integrates offline speech synthesis for efficient and responsive interaction.

The assistant is optimized to run on resource-constrained systems while maintaining high accuracy and low latency.

---
# Features
* Custom wake word detection ("Hello Airi")
* Fully offline wake word inference
* Optimized INT8 quantized machine learning models
* Synthetic and real voice training support
* Offline text-to-speech using Coqui TTS
* Docker-based deployment for Linux systems
* Low memory and CPU usage
* Modular and extensible architecture

--- 
# System Architecture
The AIRI Assistant consists of the following core components: 
```
Microphone Input
       ↓
Wake Word Detection Model (Edge Impulse)
       ↓
Command Processing Logic
       ↓
Response Generation
       ↓
Coqui TTS Synthesizer
       ↓
Audio Output
```
---
# Repository Structure
```
Airi-assistant/
│
├── coqui-tts-synthesizer/     # Text-to-speech engine
├── models-versioned/          # Trained wake word models
├── Dockerfile                 # Docker deployment configuration
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
└── .gitignore
```
# Wake Word Model Development
The wake word detection model was developed using Edge Impulse, an edge machine learning platform designed for building and deploying optimized ML models for sensor data.
## Dataset Classes
The dataset consists of three balanced classes:
* Wake Word: "Hello Airi"
* Unknown Words: unrelated speech such as "yes", "no", "left", etc.
* Noise: background sounds, silence, and environmental noise
  At least 10 minutes of wake word audio was collected to ensure reliable keyword spotting performance.
  ## Dataset Variants
  Naive Dataset
  Includes:
  * Real human recordings of "Hello Airi"
  * High variance in accents, pitch, tone, and speaking style.
  * Additional keyword spotting datasets from Edge Impulse documentation.
    This improves real-world robustness.
    ## Synthetic Dataset
    The wake word dataset was synthetically generated using the Coqui TTS synthesizer located in:
    coqui-tts-synthesizer/
    Benefits include:
    * Increased dataset scalability
    * Consistent pronunciation
    * Reduced manual data collection effort
      
      ---
      # Signal Processing and Feature Extraction
      Mel-Frequency Cepstral Coefficients (MFCC) were used for audio feature extraction.
      MFCC helps:
      * Convert raw audio into meaningful features
      * Reduce redundant information
