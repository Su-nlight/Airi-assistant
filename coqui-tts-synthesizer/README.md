# Generation of Synthetic Voice Samples
This module is made just to create multiple Synthetic Samples of a particular wake word that is to be used for an AI Assistant (named Airi in this example). 
Synthetic Audio Samples are required when you are not able to gather real data (like me who has limited number of friends) or need to create highly variance accepting Neural Network for the sake of wake word Detection.
<br> The provided code works with both Dedicated GPU and only CPU based consumer grade laptops or computers, you just need to change few things within the code to make it functional according to your need.<br>
***
## Requirements
1. <b>Laptop or computer</b> well duhh!! 
2. <b>GPU or not doesn't matter</b> : only effect will be the output will be generated slower.
3. <b>Docker Desktop</b> 
4. <b>Loads of internet with decent Speed</b> : the layers upon which this is built is heavy. (best to connect to your nosy neighbor's wifi)
5. cuda drivers (for GPU Support within docker)
6. (Optional) coffee and favorite playlist
***
## Technical Details

### Sub-directory Structure 
```
coqui-tts-synthesizer/
├── Dockerfile
├── docker-compose.yml   # Contains Support for both GPU and CPU only
├── scripts/
│   └── generate_dataset.py
├── output/              # Generated audio files (volume mounted)
└── models/              # Downloaded TTS models (volume mounted)
```

### Libraries or services used in Python Script
Following list contains those which are installed in Dockerfile.
- FFmpeg
- TTS
- scipy
- soundfile
- Torch

### How to toggle GPU suport
1. <b>Enable GPU Support</b> - You just need to keep the line 4 Uncommented (```FROM ghcr.io/coqui-ai/tts:latest```) and keep line 1 commented (```FROM ghcr.io/coqui-ai/tts-cpu:latest ```). Keep the Docker-compose file as it is, that is to keep line 7 (```runtime: nvidia```) and line 24-30 uncommented.
2. <b>CPU only Processing</b> -  invert the above scenario that is to keep Line 1 uncommented (```FROM ghcr.io/coqui-ai/tts-cpu:latest ```) and line 4 commented (```FROM ghcr.io/coqui-ai/tts:latest```). Also, we need to change the Docker-compose file, that is to keep line 7 (```runtime: nvidia```) and line 24-30 commented.

<b>Note -</b> Line 24-30 is as follows, 
```YAML
    deploy:         #comment all following lines for CPU only
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
### Enviroment variables
These key value pairs are set within the docker compose file (```docker-compose.yml```) as follows: 
```YAML
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - TTS_HOME=/app/models
      - TARGET_SAMPLES=6
      - MIN_DURATION=1.0
      - MAX_DURATION=1.3
      - BATCH_SIZE=20
      - OUTPUT_DIR=/app/output
      - WAKE_WORD=Hello Airi
      - RANDOM_SEED=42
      - FORCE_CPU=false
```
| Environment Variable | Default Value | Description |
| -------- | -------- | -------- |
| NVIDIA_VISIBLE_DEVICES | all | Row 1 Col 3 |
| TTS_HOME | `/app/models` | The directory within the container where the TTS Model will be stored. Here it is poining to the bounded volume. |
| TARGET_SAMPLES | 6 | Number of samples you need as output, that is to be found in bound output volume. You preferably want to change it as per your requirement. |
| MIN_DURATION | 1.0 | (In seconds) This is minimum length of output that you want to accept. The script rejects the voice sample whose length is less than this. |
| MAX_DURATION | 1.3 | (In seconds) This is Maximum length of output that you want to accept. The script rejects the voice sample whose length is greater than this. |
| BATCH_SIZE | 20 | Size of each Batch of output (Helps in Epoch saving) |
| OUTPUT_DIR | `/app/output` | The directory within the container where the TTS Model's output will be stored. Here it is poining to the bounded volume. |
| WAKE_WORD | Hello Airi | The wake work you want for your Purpose. You will need to change it according to your preference. |
| RANDOM_SEED | 42 | Used for reproducibility. Won't bother you much but if you want to generate samples in different Runs then change it with each consecutive run. |
| FORCE_CPU | false | This options would be helpful when you have GPU support enabled but want to force run script on CPU. |

<b>Note : </b> After changing enviroment variable(s), you would need to re-build the container (default build command = `docker-compose build` , uses cached layer for faster building).
***

## Why this approach?
### Overview of approaches I thought.
| Approach | Cost | Quality | Diversity | Setup Difficulty | Speed | Emotional Control |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Edge-TTS | Free | ⭐⭐⭐⭐ | ⭐⭐⭐ | Easy | ⭐⭐⭐⭐ Fast | ❌ None |
| Coqui TTS (Docker CPU) | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Easy | ⭐⭐ Slow | ⭐⭐⭐ Simulated |
| Coqui TTS (Docker GPU) | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐ Simulated |
| Coqui TTS (Native Windows) | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ Very Hard | ⭐⭐ Slow | ⭐⭐⭐ Simulated |
| Azure TTS API | $$$ Paid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Easy | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐⭐⭐ Native |
| Murf.ai | $$$ Paid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ No API | N/A | ⭐⭐⭐⭐ Good |
| Google Cloud TTS | $$ Paid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Easy | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐ Limited |
| ElevenLabs API | $$$ Paid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Easy | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Good |

### Advantages
1.  **No Windows Installation Hell** - Avoids dependency conflicts, build tool errors
2.  **109+ Speakers** - Maximum voice diversity for free
3.  **Completely Free** - No API costs, unlimited generation
4.  **Full Privacy** - All processing local, no data transmission
5.  **Reproducible Environment** - Works identically on any system
6.  **Persistent Storage** - Models and outputs saved to volumes
7.  **Simulated Emotions** - 9 emotional styles through audio processing
8.  **Offline Capable** - No internet after initial model download
9.  **Full Control** - Customize audio pipeline, add effects
10. **Multiple Models** - Switch between different TTS models easily

### GPU vs CPU Trade-off
Tested for 600 Samples and gives averagely following results,
- GPU: 10-50x faster (15-30 min) but setup nightmare (if cuda not installed aptly)
- CPU: Slower (2-5 hours) but works everywhere hassle-free
***
## Conclusion
This approach has various benefits helping people to create Wake word samples synthetically in different tones, accents, voices, speed, pitch, etc. while being totally local to their environment. The paid versions are truely better than this approach but for students or those who are just learning about related topics to actual objective of creating an AI assistant, while also customizing it to their needs, this approach is effortless. 