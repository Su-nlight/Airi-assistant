# -*- coding: utf-8 -*-
"""
Piper TTS Wake-Word Dataset Generator
======================================
Generates a large, acoustically diverse set of wake-word samples ("Hello Airi")
using Piper TTS (piper-tts Python package).

Architecture mirrors the Coqui / Kokoro generators in this repo:
  - Environment-variable-driven Config
  - Separated AudioProcessor
  - Batched generation loop with epoch tracking
  - GPU-accelerated ONNX inference via onnxruntime-gpu (falls back to CPU)
  - Full statistics reporting

Piper API used:
  from piper import PiperVoice, SynthesisConfig
  voice = PiperVoice.load(model_path, use_cuda=True)
  voice.synthesize_wav(text, wav_file, syn_config=syn_config)

Voice models are downloaded automatically via:
  python3 -m piper.download_voices <voice_name>

References:
  https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
  https://github.com/rhasspy/piper
"""

import io
import os
import random
import subprocess
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
class Config:
    """
    All tunables are read from environment variables so the container / runtime
    can override them without touching source code.
    """

    WAKE_WORD:      str   = os.getenv("WAKE_WORD",      "Hello Airi")
    TARGET_SAMPLES: int   = int(os.getenv("TARGET_SAMPLES",  "600"))
    MIN_DURATION:   float = float(os.getenv("MIN_DURATION",   "1.0"))
    MAX_DURATION:   float = float(os.getenv("MAX_DURATION",   "1.3"))
    BATCH_SIZE:     int   = int(os.getenv("BATCH_SIZE",       "20"))
    OUTPUT_DIR:     str   = os.getenv("OUTPUT_DIR",     "/app/output")
    MODELS_DIR:     str   = os.getenv("MODELS_DIR",     "/app/models")
    TARGET_SR:      int   = int(os.getenv("TARGET_SR",   "22050"))
    RANDOM_SEED:    int   = int(os.getenv("RANDOM_SEED",      "42"))
    FORCE_CPU:      bool  = os.getenv("FORCE_CPU", "false").lower() == "true"

    # ── Piper voice roster ────────────────────────────────────────────────────
    # Each entry is a (voice_key, onnx_filename) pair.
    # Models are downloaded on first run via `python3 -m piper.download_voices`.
    # All are English voices covering male/female, US/GB accents, low→high quality.
    # Quality tiers: x_low (16 kHz) / low (16 kHz) / medium (22.05 kHz) / high (22.05 kHz)
    VOICES: List[Dict] = [
        # American English — female
        {"key": "en_US-lessac-medium",     "onnx": "en_US-lessac-medium.onnx"},
        {"key": "en_US-libritts-high",     "onnx": "en_US-libritts-high.onnx"},
        {"key": "en_US-amy-medium",        "onnx": "en_US-amy-medium.onnx"},
        {"key": "en_US-hfc_female-medium", "onnx": "en_US-hfc_female-medium.onnx"},
        # American English — male
        {"key": "en_US-ryan-medium",       "onnx": "en_US-ryan-medium.onnx"},
        {"key": "en_US-hfc_male-medium",   "onnx": "en_US-hfc_male-medium.onnx"},
        {"key": "en_US-joe-medium",        "onnx": "en_US-joe-medium.onnx"},
        # British English — mixed
        {"key": "en_GB-alan-medium",       "onnx": "en_GB-alan-medium.onnx"},
        {"key": "en_GB-jenny_dioco-medium","onnx": "en_GB-jenny_dioco-medium.onnx"},
        {"key": "en_GB-vctk-medium",       "onnx": "en_GB-vctk-medium.onnx"},
    ]

    # ── Synthesis variation ranges ────────────────────────────────────────────
    # length_scale: >1 = slower, <1 = faster  (Piper's speed control)
    LENGTH_SCALE_RANGE: List[float] = [0.80, 0.90, 1.00, 1.10, 1.20]
    # noise_scale: controls audio variation (stochastic sampling)
    NOISE_SCALE_RANGE:  List[float] = [0.333, 0.500, 0.667, 0.800]
    # noise_w_scale: controls speaking-style variation (duration stochasticity)
    NOISE_W_RANGE:      List[float] = [0.6, 0.8, 1.0]
    # Post-synthesis pitch shift in semitones (librosa)
    PITCH_SHIFT_RANGE:  List[int]   = [-3, -2, -1, 0, 1, 2, 3]

    # ── Emotion simulation (amplitude + optional tremolo) ─────────────────────
    # Mirrors the approach used in the Coqui and Kokoro generators.
    EMOTION_STYLES: Dict[str, float] = {
        "neutral":   1.00,
        "happy":     1.10,
        "excited":   1.15,
        "calm":      0.88,
        "sad":       0.85,
        "angry":     1.20,
        "fearful":   1.05,   # tremolo applied separately
        "disgusted": 0.95,
        "surprised": 1.08,
    }


# ──────────────────────────────────────────────
#  AUDIO PROCESSOR
# ──────────────────────────────────────────────
class AudioProcessor:
    """
    Stateless audio transformations applied after Piper synthesis.
    All methods operate on float32 numpy arrays normalised to ±1.
    """

    @staticmethod
    def int16_bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        """Convert raw PCM int16 bytes → float32 numpy array in [-1, 1]."""
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    @staticmethod
    def apply_emotion(audio: np.ndarray, emotion: str, native_sr: int) -> np.ndarray:
        """Scale amplitude and optionally add tremolo effect."""
        scalar = Config.EMOTION_STYLES.get(emotion, 1.00)
        audio = audio * scalar

        if emotion == "fearful":
            t = np.arange(len(audio))
            tremolo = 1.0 + 0.08 * np.sin(2 * np.pi * 5 * t / native_sr)
            audio = audio * tremolo

        # Normalise to ±0.95 to prevent clipping after stacking effects
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95
        return audio

    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """Pitch-shift by n_steps semitones using librosa (CPU only)."""
        if n_steps == 0:
            return audio
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample to target_sr if needed."""
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def fit_duration(
        audio: np.ndarray, sr: int,
        min_dur: float, max_dur: float
    ) -> Optional[np.ndarray]:
        """
        Time-stretch to window midpoint if audio falls outside [min_dur, max_dur].
        Returns None only when the required stretch ratio is pathological (>2× or <0.5×).
        """
        duration = len(audio) / sr
        if min_dur <= duration <= max_dur:
            return audio

        target = (min_dur + max_dur) / 2.0
        ratio  = duration / target          # >1 = too long, <1 = too short
        if not (0.5 <= ratio <= 2.0):
            return None                     # genuinely out-of-range — discard

        return librosa.effects.time_stretch(audio, rate=ratio)


# ──────────────────────────────────────────────
#  VOICE MODEL MANAGER
# ──────────────────────────────────────────────
class VoiceModelManager:
    """
    Downloads Piper ONNX models on first run and loads them into memory.
    Keeps one PiperVoice object per model to avoid repeated disk I/O.
    """

    def __init__(self, models_dir: Path, use_cuda: bool):
        self.models_dir = models_dir
        self.use_cuda   = use_cuda
        self._cache: Dict[str, object] = {}   # voice_key → PiperVoice

    def get_voice(self, voice_info: Dict):
        """Return a cached (or newly loaded) PiperVoice for the given voice entry."""
        key       = voice_info["key"]
        onnx_name = voice_info["onnx"]

        if key in self._cache:
            return self._cache[key]

        model_path = self.models_dir / onnx_name

        # Download if not already present
        if not model_path.exists():
            self._download(key, model_path)

        if not model_path.exists():
            print(f"  ⚠  Model file not found after download attempt: {model_path}")
            return None

        try:
            # Import here so the container can start even before piper is fully ready
            from piper import PiperVoice
            voice = PiperVoice.load(str(model_path), use_cuda=self.use_cuda)
            self._cache[key] = voice
            print(f"  ✓  Loaded voice: {key}  (SR={voice.config.sample_rate} Hz)")
            return voice
        except Exception as exc:
            print(f"  ⚠  Failed to load {key}: {exc}")
            return None

    def _download(self, voice_key: str, target_path: Path):
        """Download a Piper voice model to models_dir using the built-in CLI."""
        print(f"  ⬇  Downloading voice model: {voice_key} …")
        try:
            subprocess.run(
                [
                    "python3", "-m", "piper.download_voices",
                    voice_key,
                    "--data-dir", str(self.models_dir),
                ],
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  ⚠  Download failed for {voice_key}: {exc}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠  Download timed out for {voice_key}")


# ──────────────────────────────────────────────
#  DATASET GENERATOR
# ──────────────────────────────────────────────
class PiperDatasetGenerator:

    def __init__(self, config: Config):
        self.cfg        = config
        self.output_dir = Path(config.OUTPUT_DIR)
        self.models_dir = Path(config.MODELS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        # Resolve CUDA availability
        self.use_cuda = self._resolve_device()

        self.audio_proc   = AudioProcessor()
        self.voice_mgr    = VoiceModelManager(self.models_dir, self.use_cuda)

        # Pre-load all voice models so download happens up front
        self._preload_voices()

        # Statistics
        self.valid_count    = 0
        self.rejected_count = 0
        self.total_attempts = 0

    # ── Device resolution ────────────────────────────────────────────────────

    def _resolve_device(self) -> bool:
        """Return True if CUDA should be used, False for CPU."""
        if self.cfg.FORCE_CPU:
            return False
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            cuda_ok   = "CUDAExecutionProvider" in providers
            if cuda_ok:
                print("  🚀 CUDA detected — GPU inference enabled.")
            else:
                print("  ⚠  CUDA not available — falling back to CPU.")
            return cuda_ok
        except Exception:
            print("  ⚠  Could not query onnxruntime providers — using CPU.")
            return False

    # ── Voice pre-loading ────────────────────────────────────────────────────

    def _preload_voices(self):
        """Download and load all configured voice models."""
        print(f"\n{'='*62}")
        print("  Downloading & loading Piper voice models …")
        print(f"{'='*62}")
        available = []
        for v in self.cfg.VOICES:
            voice = self.voice_mgr.get_voice(v)
            if voice is not None:
                available.append(v)
        self.available_voices = available
        print(f"\n  {len(self.available_voices)}/{len(self.cfg.VOICES)} voices loaded.\n")

        if not self.available_voices:
            raise RuntimeError("No voice models could be loaded. Check network / models dir.")

    # ── Sample-level helpers ─────────────────────────────────────────────────

    def _random_params(self) -> Dict:
        return {
            "voice_info":   random.choice(self.available_voices),
            "length_scale": random.choice(self.cfg.LENGTH_SCALE_RANGE),
            "noise_scale":  random.choice(self.cfg.NOISE_SCALE_RANGE),
            "noise_w":      random.choice(self.cfg.NOISE_W_RANGE),
            "pitch_steps":  random.choice(self.cfg.PITCH_SHIFT_RANGE),
            "emotion":      random.choice(list(self.cfg.EMOTION_STYLES.keys())),
        }

    def _generate_single(self, params: Dict, output_path: Path) -> bool:
        """
        Synthesise one utterance with Piper, apply post-processing, check duration,
        and write to disk.  Returns True on acceptance, False on rejection.
        """
        try:
            from piper import PiperVoice, SynthesisConfig

            voice = self.voice_mgr.get_voice(params["voice_info"])
            if voice is None:
                return False

            native_sr = voice.config.sample_rate   # 16000 or 22050 Hz

            # ── Piper synthesis → in-memory WAV ──────────────────────────────
            # SynthesisConfig controls Piper-level variation knobs.
            syn_cfg = SynthesisConfig(
                length_scale=params["length_scale"],   # speaking rate
                noise_scale=params["noise_scale"],     # audio stochasticity
                noise_w_scale=params["noise_w"],       # duration stochasticity
                volume=1.0,
                normalize_audio=True,
            )

            # synthesize_wav writes PCM into the wave.Wave_write object.
            # We capture into a BytesIO buffer to avoid writing temp files.
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)          # int16
                wav_out.setframerate(native_sr)
                voice.synthesize_wav(
                    self.cfg.WAKE_WORD,
                    wav_out,
                    syn_config=syn_cfg,
                )

            # ── Decode PCM bytes → float32 numpy ─────────────────────────────
            buf.seek(0)
            with wave.open(buf, "rb") as wav_in:
                raw_bytes = wav_in.readframes(wav_in.getnframes())

            audio = self.audio_proc.int16_bytes_to_float32(raw_bytes)

            # ── Pitch shift (librosa, CPU) ────────────────────────────────────
            audio = self.audio_proc.pitch_shift(
                audio, native_sr, params["pitch_steps"]
            )

            # ── Emotion simulation (amplitude + optional tremolo) ─────────────
            audio = self.audio_proc.apply_emotion(
                audio, params["emotion"], native_sr
            )

            # ── Resample to target SR if needed ──────────────────────────────
            audio = self.audio_proc.resample(
                audio, native_sr, self.cfg.TARGET_SR
            )

            # ── Duration gate with time-stretch fallback ──────────────────────
            audio = self.audio_proc.fit_duration(
                audio, self.cfg.TARGET_SR,
                self.cfg.MIN_DURATION, self.cfg.MAX_DURATION,
            )
            if audio is None:
                return False

            # ── Write accepted sample ─────────────────────────────────────────
            sf.write(str(output_path), audio, self.cfg.TARGET_SR, subtype="PCM_16")
            return True

        except Exception as exc:
            print(f"    ⚠  Generation error: {exc}")
            if output_path.exists():
                output_path.unlink()
            return False

    # ── Batch / epoch loop ───────────────────────────────────────────────────

    def _run_batch(self, epoch: int, batch_size: int) -> int:
        print(f"\n{'─'*62}")
        print(f"  EPOCH {epoch:>3} │ batch_size={batch_size}  "
              f"│ valid so far={self.valid_count}/{self.cfg.TARGET_SAMPLES}")
        print(f"{'─'*62}")

        batch_valid    = 0
        batch_rejected = 0

        for i in range(batch_size):
            params = self._random_params()
            self.total_attempts += 1

            voice_key = params["voice_info"]["key"]
            ts        = int(time.time() * 1000)
            filename  = (
                f"airi_{self.valid_count + batch_valid:04d}_"
                f"{voice_key.replace('/', '_')}_"
                f"ls{params['length_scale']:.1f}_"
                f"pit{params['pitch_steps']:+d}_"
                f"{params['emotion'][:4]}_"
                f"{ts}.wav"
            )
            output_path = self.output_dir / filename

            ok = self._generate_single(params, output_path)

            tag = voice_key.split("-")[0]   # e.g. "en_US"
            if ok:
                batch_valid += 1
                print(
                    f"  ✓ [{i+1:>3}/{batch_size}] "
                    f"voice={voice_key:<30} "
                    f"ls={params['length_scale']:.1f}  "
                    f"pit={params['pitch_steps']:+d}  "
                    f"emo={params['emotion']}"
                )
            else:
                batch_rejected += 1
                print(
                    f"  ✗ [{i+1:>3}/{batch_size}] REJECTED — "
                    f"duration outside [{self.cfg.MIN_DURATION},{self.cfg.MAX_DURATION}]s"
                )

        self.valid_count    += batch_valid
        self.rejected_count += batch_rejected

        print(f"\n  Batch summary → valid: {batch_valid}  rejected: {batch_rejected}")
        print(f"  Cumulative    → valid: {self.valid_count}  "
              f"rejected: {self.rejected_count}  attempts: {self.total_attempts}")
        return batch_valid

    # ── Public entry-point ───────────────────────────────────────────────────

    def generate_dataset(self):
        print("\n" + "="*62)
        print("  PIPER TTS WAKE-WORD DATASET GENERATOR")
        print("="*62)
        print(f"  Wake word      : '{self.cfg.WAKE_WORD}'")
        print(f"  Target samples : {self.cfg.TARGET_SAMPLES}")
        print(f"  Duration range : {self.cfg.MIN_DURATION}s – {self.cfg.MAX_DURATION}s")
        print(f"  Output SR      : {self.cfg.TARGET_SR} Hz")
        print(f"  Batch size     : {self.cfg.BATCH_SIZE}")
        print(f"  Output dir     : {self.output_dir.absolute()}")
        print(f"  Models dir     : {self.models_dir.absolute()}")
        print(f"  Inference      : {'CUDA (GPU)' if self.use_cuda else 'CPU'}")
        print(f"  Voices loaded  : {len(self.available_voices)}")
        print(f"  Emotion styles : {len(self.cfg.EMOTION_STYLES)}")
        print("="*62)

        if not self.use_cuda:
            print("\n  ⏱  CPU mode — inference is fast with Piper (~0.1–0.3s/sample)")
        else:
            print("\n  ⚡ GPU mode — even faster inference enabled")

        epoch      = 1
        start_time = time.time()

        while self.valid_count < self.cfg.TARGET_SAMPLES:
            remaining  = self.cfg.TARGET_SAMPLES - self.valid_count
            batch_size = min(self.cfg.BATCH_SIZE, remaining)
            self._run_batch(epoch, batch_size)
            epoch += 1

        elapsed = time.time() - start_time
        success_rate = (
            self.valid_count / self.total_attempts * 100
            if self.total_attempts > 0 else 0.0
        )

        print("\n" + "="*62)
        print("  DATASET GENERATION COMPLETE")
        print("="*62)
        print(f"  Valid samples  : {self.valid_count}")
        print(f"  Rejected       : {self.rejected_count}")
        print(f"  Total attempts : {self.total_attempts}")
        print(f"  Success rate   : {success_rate:.1f}%")
        print(f"  Elapsed time   : {elapsed:.1f}s")
        print(f"  Avg / sample   : {elapsed / max(self.valid_count, 1):.2f}s")
        print(f"  Output dir     : {self.output_dir.absolute()}")
        print("="*62 + "\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg       = Config()
    generator = PiperDatasetGenerator(cfg)
    generator.generate_dataset()