# -*- coding: utf-8 -*-
"""
Kokoro TTS Wake-Word Dataset Generator
Production-grade rewrite of kokorov1.py

Architecture mirrors the Coqui TTS generator:
  - Environment-variable-driven Config
  - Separated AudioProcessor
  - Batched generation loop with epoch tracking
  - GPU-accelerated Kokoro inference (falls back to CPU gracefully)
  - Full statistics reporting
"""

import os
import time
import random
import warnings
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import soundfile as sf
import librosa
from kokoro import KPipeline

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
class Config:
    """
    All tunables are read from environment variables so the container/runtime
    can override them without touching source code. Mirrors Coqui's approach.
    """

    WAKE_WORD:       str   = os.getenv("WAKE_WORD",       "Hello Airi")
    TARGET_SAMPLES:  int   = int(os.getenv("TARGET_SAMPLES",  "15"))
    MIN_DURATION:    float = float(os.getenv("MIN_DURATION",  "1.0"))
    MAX_DURATION:    float = float(os.getenv("MAX_DURATION",  "1.3"))
    BATCH_SIZE:      int   = int(os.getenv("BATCH_SIZE",      "20"))
    OUTPUT_DIR:      str   = os.getenv("OUTPUT_DIR",      "/app/output")
    RANDOM_SEED:     int   = int(os.getenv("RANDOM_SEED",     "42"))
    FORCE_CPU:       bool  = os.getenv("FORCE_CPU", "false").lower() == "true"

    # Kokoro runtime settings
    LANG_CODE:       str   = os.getenv("LANG_CODE",       "a")   # 'a' = American English
    NATIVE_SR:       int   = 24_000   # Kokoro's fixed output sample rate
    TARGET_SR:       int   = int(os.getenv("TARGET_SR",   "16000"))  # normalised output SR

    # Voice roster — all voices shipped with kokoro>=0.1.9
    VOICES = [
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam",  "am_michael",
        "bf_emma",  "bf_isabella",
        "bm_lewis", "bm_george",
    ]

    # Variation ranges
    SPEED_RANGE       = [0.8, 0.9, 1.0, 1.1, 1.2]
    PITCH_SHIFT_RANGE = [ -2, -1, 0, 1, 2, 3]   # semitones

    # Emotion tag → energy scalar (mirrors Coqui's emotion simulation approach)
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

    @staticmethod
    def get_device() -> str:
        if Config.FORCE_CPU:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
#  AUDIO PROCESSOR
# ──────────────────────────────────────────────
class AudioProcessor:
    """
    Stateless audio transformations.
    All heavy DSP (pitch shift, resample) lives here, keeping the generator
    class focused on orchestration — same split as the Coqui implementation.
    """

    @staticmethod
    def apply_emotion(audio: np.ndarray, emotion: str) -> np.ndarray:
        """Scale amplitude and optionally add tremolo to simulate emotion."""
        scalar = Config.EMOTION_STYLES.get(emotion, 1.0)
        audio = audio * scalar

        if emotion == "fearful":
            t = np.arange(len(audio))
            tremolo = 1.0 + 0.08 * np.sin(2 * np.pi * 5 * t / Config.NATIVE_SR)
            audio = audio * tremolo

        # Normalise to ±0.95  (prevents clipping after stacking effects)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95
        return audio

    @staticmethod
    def pitch_shift(audio: np.ndarray, n_steps: float, sr: int) -> np.ndarray:
        """Librosa pitch shift — runs on CPU regardless of main device."""
        if n_steps == 0.0:
            return audio
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    @staticmethod
    def fit_duration(audio: np.ndarray, sr: int,
                    min_dur: float, max_dur: float) -> Optional[np.ndarray]:
        """
        Time-stretch audio to the window midpoint if it falls outside bounds.
        Returns None only if the required stretch ratio is pathological (>2.0x).
        """
        duration = len(audio) / sr
        target   = (min_dur + max_dur) / 2.0      # e.g. 1.15s

        if min_dur <= duration <= max_dur:
            return audio                           # already in window, no-op

        ratio = duration / target                  # >1 = too long, <1 = too short
        if not (0.5 <= ratio <= 2.0):
            return None                            # genuinely pathological, discard

        return librosa.effects.time_stretch(audio, rate=ratio)


# ──────────────────────────────────────────────
#  DATASET GENERATOR
# ──────────────────────────────────────────────
class KokoroDatasetGenerator:

    def __init__(self, config: Config):
        self.cfg = config
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)

        self.device = Config.get_device()
        self.audio_proc = AudioProcessor()

        # ── Kokoro pipeline initialisation ──────────────────────────────────
        print(f"\n{'='*62}")
        print("  Initialising Kokoro TTS Pipeline")
        print(f"  Device : {self.device.upper()}")
        if self.device == "cpu":
            print("  ⚠  CPU mode — inference will be slower.")
            print("  💡 Set FORCE_CPU=false and ensure CUDA drivers are present")
            print("     for GPU-accelerated inference.")
        else:
            print("  🚀 GPU detected — fast inference enabled.")
        print(f"{'='*62}\n")

        try:
            # KPipeline accepts a `device` kwarg; pass the resolved device so
            # all Kokoro model weights are loaded onto the correct accelerator.
            self.pipeline = KPipeline(
                lang_code=config.LANG_CODE,
                device=self.device,
            )
            print(f"  Pipeline ready on {self.device.upper()}.")
        except TypeError:
            # Older kokoro builds (<0.2) don't expose `device` in __init__.
            # Fall back silently; torch will still prefer CUDA if available.
            print("  ℹ  Kokoro version does not accept explicit device arg.")
            print("     Falling back to default device detection.")
            self.pipeline = KPipeline(lang_code=config.LANG_CODE)

        # ── Statistics counters ──────────────────────────────────────────────
        self.valid_count    = 0
        self.rejected_count = 0
        self.total_attempts = 0

        # existing line (already in __init__)
        self.audio_proc = AudioProcessor()

        # ADD these two lines directly after:
        self.valid_speeds = self._calibrate_speeds()

        # Statistics counters (already present)
        self.valid_count    = 0

    # ── Sample-level helpers ─────────────────────────────────────────────────

    def _random_params(self) -> Dict:
        return {
            "voice":       random.choice(self.cfg.VOICES),
            "speed":       random.choice(self.valid_speeds),       # ← calibrated list
            "pitch_steps": random.choice(self.cfg.PITCH_SHIFT_RANGE),
            "emotion":     random.choice(list(self.cfg.EMOTION_STYLES.keys())),
        }

    def _generate_single(self, params: Dict, output_path: Path) -> bool:
        """
        Synthesise one utterance, apply post-processing, duration-filter,
        and write to disk. Returns True on acceptance, False on rejection.
        """
        try:
            gen = self.pipeline(
                self.cfg.WAKE_WORD,
                voice=params["voice"],
                speed=params["speed"],
                split_pattern=None,         # treat entire string as one segment
            )

            for _, _, audio_tensor in gen:
                # ── tensor → numpy (float32, shape [T]) ─────────────────────
                audio: np.ndarray = audio_tensor.cpu().numpy().astype(np.float32)

                # ── pitch shift (CPU-only librosa call) ──────────────────────
                audio = self.audio_proc.pitch_shift(
                    audio, params["pitch_steps"], sr=self.cfg.NATIVE_SR
                )

                # ── emotion amplitude shaping ────────────────────────────────
                audio = self.audio_proc.apply_emotion(audio, params["emotion"])

                # ── resample to target SR ────────────────────────────────────
                audio = self.audio_proc.resample(
                    audio,
                    orig_sr=self.cfg.NATIVE_SR,
                    target_sr=self.cfg.TARGET_SR,
                )

                # ── duration gate ────────────────────────────────────────────
                duration = len(audio) / self.cfg.TARGET_SR
                audio = self.audio_proc.fit_duration(
                    audio, self.cfg.TARGET_SR,
                    self.cfg.MIN_DURATION, self.cfg.MAX_DURATION
                )
                if audio is None:
                    return False

                sf.write(str(output_path), audio, self.cfg.TARGET_SR)
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

            ts       = int(time.time() * 1000)
            filename = (
                f"airi_{self.valid_count + batch_valid:04d}_"
                f"{params['voice']}_"
                f"spd{params['speed']:.1f}_"
                f"pit{params['pitch_steps']:+d}_"
                f"{params['emotion'][:4]}_"
                f"{ts}.wav"
            )
            output_path = self.output_dir / filename

            ok = self._generate_single(params, output_path)

            if ok:
                batch_valid += 1
                print(
                    f"  ✓ [{i+1:>3}/{batch_size}] voice={params['voice']:<12} "
                    f"spd={params['speed']:.1f}  pit={params['pitch_steps']:+d}  "
                    f"emo={params['emotion']}"
                )
            else:
                batch_rejected += 1
                print(
                    f"  ✗ [{i+1:>3}/{batch_size}] REJECTED — duration out of "
                    f"[{self.cfg.MIN_DURATION},{self.cfg.MAX_DURATION}]s"
                )

        self.valid_count    += batch_valid
        self.rejected_count += batch_rejected

        print(f"\n  Batch summary → valid: {batch_valid}  rejected: {batch_rejected}")
        print(f"  Cumulative    → valid: {self.valid_count}  "
              f"rejected: {self.rejected_count}  attempts: {self.total_attempts}")
        return batch_valid

    def _calibrate_speeds(self) -> list:
        """
        Synthesise the wake word once per speed value at neutral pitch/emotion.
        Keep only speeds whose raw output duration falls within 2x the window
        (time_stretch handles the rest).
        """
        print("  🔧 Calibrating speed range...")
        valid_speeds = []
        target = (self.cfg.MIN_DURATION + self.cfg.MAX_DURATION) / 2.0

        for speed in self.cfg.SPEED_RANGE:
            gen = self.pipeline(self.cfg.WAKE_WORD, voice="af_heart",
                                speed=speed, split_pattern=None)
            for _, _, t in gen:
                audio = t.cpu().numpy()
                audio = self.audio_proc.resample(
                    audio, self.cfg.NATIVE_SR, self.cfg.TARGET_SR)
                dur   = len(audio) / self.cfg.TARGET_SR
                ratio = dur / target
                if 0.5 <= ratio <= 2.0:          # time_stretch can handle this
                    valid_speeds.append(speed)
                    print(f"    speed={speed:.1f} → {dur:.3f}s ✓")
                else:
                    print(f"    speed={speed:.1f} → {dur:.3f}s ✗ (ratio {ratio:.2f} — excluded)")
                break

        print(f"  Valid speeds: {valid_speeds}\n")
        return valid_speeds if valid_speeds else self.cfg.SPEED_RANGE


    # ── Public entry-point ───────────────────────────────────────────────────

    def generate_dataset(self):
        print("\n" + "="*62)
        print("  KOKORO WAKE-WORD DATASET GENERATOR")
        print("="*62)
        print(f"  Wake word      : '{self.cfg.WAKE_WORD}'")
        print(f"  Target samples : {self.cfg.TARGET_SAMPLES}")
        print(f"  Duration range : {self.cfg.MIN_DURATION}s – {self.cfg.MAX_DURATION}s")
        print(f"  Output SR      : {self.cfg.TARGET_SR} Hz")
        print(f"  Batch size     : {self.cfg.BATCH_SIZE}")
        print(f"  Output dir     : {self.output_dir.absolute()}")
        print(f"  Device         : {self.device.upper()}")
        print(f"  Voices         : {len(self.cfg.VOICES)}")
        print(f"  Emotion styles : {len(self.cfg.EMOTION_STYLES)}")
        print("="*62)

        if self.device == "cpu":
            print("\n  ⏱  CPU mode — estimated ~2–4 h for 600 samples")
        else:
            print("\n  ⚡ GPU mode — estimated ~10–20 min for 600 samples")

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
        print(f"  Avg / sample   : {elapsed/max(self.valid_count,1):.2f}s")
        print(f"  Output dir     : {self.output_dir.absolute()}")
        print("="*62 + "\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg       = Config()
    generator = KokoroDatasetGenerator(cfg)
    generator.generate_dataset()