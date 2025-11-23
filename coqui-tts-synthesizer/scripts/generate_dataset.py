import os
import random
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch
from TTS.api import TTS
import soundfile as sf
from scipy import signal

# ===== CONFIGURATION =====
class Config:
    TARGET_SAMPLES = int(os.getenv("TARGET_SAMPLES", "600"))
    MIN_DURATION = float(os.getenv("MIN_DURATION", "1.0"))
    MAX_DURATION = float(os.getenv("MAX_DURATION", "1.3"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
    TEXT = os.getenv("WAKE_WORD", "Hello Airi")
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    # TTS Model settings
    MODEL_NAME = "tts_models/en/vctk/vits"  # Multi-speaker English model
    
    # Auto-detect device
    @staticmethod
    def get_device():
        if Config.FORCE_CPU:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    DEVICE = None  # Will be set after initialization
    
    # Voice variations
    EMOTION_STYLES = [
        "neutral", "happy", "sad", "angry", "surprised",
        "fearful", "disgusted", "calm", "excited"
    ]
    
    SPEED_RANGE = [0.8, 0.9, 1.0, 1.1, 1.2]
    PITCH_SHIFT_RANGE = [-3, -2, -1, 0, 1, 2, 3]  # Semitones


# ===== AUDIO PROCESSING =====
class AudioProcessor:
    """Handles audio transformations"""
    
    @staticmethod
    def change_pitch(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """Shift pitch by n_steps semitones"""
        if n_steps == 0:
            return audio
        
        # Calculate the frequency ratio
        ratio = 2 ** (n_steps / 12.0)
        
        # Resample to change pitch
        new_length = int(len(audio) / ratio)
        resampled = signal.resample(audio, new_length)
        
        # Resample back to original length (changes pitch but not duration)
        result = signal.resample(resampled, len(audio))
        return result
    
    @staticmethod
    def change_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Change speed of audio"""
        if speed == 1.0:
            return audio
        
        indices = np.round(np.arange(0, len(audio), speed)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]
    
    @staticmethod
    def add_emotion_simulation(audio: np.ndarray, emotion: str) -> np.ndarray:
        """Simulate emotions through audio processing"""
        if emotion == "happy" or emotion == "excited":
            # Slightly increase energy
            audio = audio * 1.1
        elif emotion == "sad" or emotion == "calm":
            # Decrease energy slightly
            audio = audio * 0.9
        elif emotion == "angry":
            # Increase energy more
            audio = audio * 1.2
        elif emotion == "fearful":
            # Add slight tremolo effect
            tremolo = 1 + 0.1 * np.sin(2 * np.pi * 5 * np.arange(len(audio)) / 22050)
            audio = audio * tremolo
        
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio


# ===== COQUI DATASET GENERATOR =====
class CoquiDatasetGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize random seed
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # Set device
        Config.DEVICE = Config.get_device()
        
        # Initialize TTS model
        print(f"\n{'='*60}")
        print(f"Initializing Coqui TTS...")
        print(f"Device: {Config.DEVICE.upper()}")
        if Config.DEVICE == "cpu":
            print("⚠️  Running on CPU - This will be slower.")
            print("💡 For faster generation, use GPU version (docker-compose-gpu.yml)")
        else:
            print("🚀 GPU detected - Fast generation enabled!")
        print(f"{'='*60}\n")
        
        print(f"Loading TTS model: {config.MODEL_NAME}...")
        
        try:
            self.tts = TTS(model_name=config.MODEL_NAME).to(Config.DEVICE)
        except Exception as e:
            print(f"⚠️  Error loading model on {Config.DEVICE}: {e}")
            print("Falling back to CPU...")
            Config.DEVICE = "cpu"
            self.tts = TTS(model_name=config.MODEL_NAME).to("cpu")
        
        # Get available speakers
        self.speakers = self.tts.speakers if hasattr(self.tts, 'speakers') else []
        print(f"Available speakers: {len(self.speakers)}")
        
        # Audio processor
        self.audio_processor = AudioProcessor()
        
        # Statistics
        self.total_generated = 0
        self.valid_samples = 0
        self.rejected_samples = 0
    
    def generate_sample_config(self) -> Dict:
        """Generate random configuration for a single sample"""
        speaker = random.choice(self.speakers) if self.speakers else None
        
        return {
            "speaker": speaker,
            "emotion": random.choice(self.config.EMOTION_STYLES),
            "speed": random.choice(self.config.SPEED_RANGE),
            "pitch_shift": random.choice(self.config.PITCH_SHIFT_RANGE),
        }
    
    def generate_single_audio(
        self,
        text: str,
        speaker: str,
        emotion: str,
        speed: float,
        pitch_shift: int,
        output_path: Path
    ) -> bool:
        """Generate a single audio file"""
        try:
            # Generate base audio
            if speaker:
                wav = self.tts.tts(text=text, speaker=speaker)
            else:
                wav = self.tts.tts(text=text)
            
            # Convert to numpy array
            audio = np.array(wav)
            
            # Apply speed change
            audio = self.audio_processor.change_speed(audio, speed)
            
            # Apply pitch shift
            audio = self.audio_processor.change_pitch(audio, 22050, pitch_shift)
            
            # Apply emotion simulation
            audio = self.audio_processor.add_emotion_simulation(audio, emotion)
            
            # Check duration
            duration = len(audio) / 22050.0
            
            if self.config.MIN_DURATION <= duration <= self.config.MAX_DURATION:
                # Save audio
                sf.write(str(output_path), audio, 22050)
                return True
            else:
                return False
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def generate_batch(self, batch_num: int, batch_size: int) -> int:
        """Generate a batch of audio samples"""
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}: Generating {batch_size} samples...")
        print(f"{'='*60}")
        
        batch_valid = 0
        batch_rejected = 0
        
        for i in range(batch_size):
            config = self.generate_sample_config()
            
            # Generate filename
            timestamp = int(time.time() * 1000)
            speaker_id = config['speaker'][:10] if config['speaker'] else "nospeaker"
            filename = (
                f"airi_{timestamp}_{i}_"
                f"{speaker_id}_"
                f"{config['emotion'][:4]}_"
                f"spd{config['speed']:.1f}_"
                f"pit{config['pitch_shift']:+d}.wav"
            )
            
            output_path = self.output_dir / filename
            
            # Generate audio
            success = self.generate_single_audio(
                text=self.config.TEXT,
                speaker=config['speaker'],
                emotion=config['emotion'],
                speed=config['speed'],
                pitch_shift=config['pitch_shift'],
                output_path=output_path
            )
            
            if success:
                batch_valid += 1
                print(f"  ✓ Sample {i+1}: speaker={config['speaker'][:15] if config['speaker'] else 'none'} "
                      f"emotion={config['emotion']} speed={config['speed']:.1f} pitch={config['pitch_shift']:+d}")
            else:
                batch_rejected += 1
                print(f"  ✗ Sample {i+1}: Duration out of range (rejected)")
        
        self.valid_samples += batch_valid
        self.rejected_samples += batch_rejected
        self.total_generated += batch_size
        
        print(f"\nBatch {batch_num} Complete:")
        print(f"  Valid: {batch_valid}, Rejected: {batch_rejected}")
        print(f"  Total Valid So Far: {self.valid_samples}/{self.config.TARGET_SAMPLES}")
        
        return batch_valid
    
    def generate_dataset(self):
        """Main generation loop"""
        print("\n" + "="*60)
        print("AIRI VOICE DATASET GENERATOR (COQUI TTS)")
        print("="*60)
        print(f"Target Samples: {self.config.TARGET_SAMPLES}")
        print(f"Text: '{self.config.TEXT}'")
        print(f"Duration Range: {self.config.MIN_DURATION}s - {self.config.MAX_DURATION}s")
        print(f"Output Directory: {self.output_dir}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Device: {Config.DEVICE.upper()}")
        print(f"Model: {self.config.MODEL_NAME}")
        print(f"Available Speakers: {len(self.speakers)}")
        print("="*60)
        
        if Config.DEVICE == "cpu":
            print("\n⏱️  CPU Mode: Estimated time ~2-5 hours for 600 samples")
        else:
            print("\n⚡ GPU Mode: Estimated time ~15-30 minutes for 600 samples")
        
        epoch = 1
        start_time = time.time()
        
        while self.valid_samples < self.config.TARGET_SAMPLES:
            remaining = self.config.TARGET_SAMPLES - self.valid_samples
            batch_size = min(self.config.BATCH_SIZE, remaining)
            
            print(f"\n--- EPOCH {epoch} ---")
            print(f"Remaining samples needed: {remaining}")
            
            self.generate_batch(epoch, batch_size)
            
            epoch += 1
        
        elapsed_time = time.time() - start_time
        
        # Final statistics
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"Total Valid Samples: {self.valid_samples}")
        print(f"Total Rejected: {self.rejected_samples}")
        print(f"Total Generated: {self.total_generated}")
        print(f"Success Rate: {self.valid_samples/self.total_generated*100:.1f}%")
        print(f"Time Elapsed: {elapsed_time:.1f}s")
        print(f"Avg Time per Sample: {elapsed_time/self.valid_samples:.2f}s")
        print(f"Output Directory: {self.output_dir.absolute()}")
        print("="*60)


# ===== MAIN =====
if __name__ == "__main__":
    config = Config()
    generator = CoquiDatasetGenerator(config)
    generator.generate_dataset()