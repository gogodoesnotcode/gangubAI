"""Configuration optimizations for latency-sensitive voice pipeline.

Key tuning parameters for Raspberry Pi and embedded systems:
- SILENCE_DURATION_S: Lower for faster turn cutoff (0.7-1.0 recommended)
- WHISPER_CPP_THREADS: Test empirically (2,4,6,8) - more threads != faster on ARM
- Whisper flags: no-timestamps, no-verbose for faster execution
"""

import os

# ── Recording ──────────────────────────────────────────────
INPUT_DEVICE_NAME: str | None = None
SILENCE_THRESHOLD: float = 0.006

# LATENCY TUNING: Reduce from 1.5s to 0.7-1.0 for faster turn cutoff
# Trade-off: Lower value = faster response, but may cut off slow speech
SILENCE_DURATION_S: float = 1.0  # Tuned for responsiveness

MAX_RECORD_TIME_S: float = 30.0
PRE_RECORD_DELAY_S: float = 0.4
TTS_USE_APLAY: bool = True
RECORDING_OUTPUT_FILE: str = "gangubai_input.wav"


# -- Wake word (OpenWakeWord) ---------------------------------------------
WAKE_WORD_MODEL: str = "./wakeword_models/hi_gungu_bai.onnx"
WAKE_WORD_ASSETS_DIR: str = "./wakeword_models"
WAKE_WORD_THRESHOLD: float = 0.5
OWW_SAMPLE_RATE: int = 16000
OWW_CHUNK_SIZE: int = 1280
ENABLE_KEYBOARD_PTT: bool = True


# -- Transcription (whisper.cpp) - OPTIMIZED FOR LATENCY ----
WHISPER_CPP_BIN: str = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_CPP_MODEL: str = "./whisper.cpp/models/ggml-base.en.bin"
WHISPER_LANGUAGE: str = "en"

# LATENCY TUNING: Test 2,4,6,8 threads on your Pi
# More threads ≠ faster on ARM SBCs due to contention
# Recommended: Start with 4, benchmark with: python3 -m ai.voice.transcriber_persistent benchmark.wav --benchmark
WHISPER_CPP_THREADS: int = 4

# Timeout (should rarely trigger on modern hardware)
WHISPER_TIMEOUT_S: int = 30

# OPTIMIZATION FLAGS (for whisper.cpp)
# These are passed when running transcription
WHISPER_NO_TIMESTAMPS: bool = True  # Disable timestamp generation (~5-10% speedup)
WHISPER_NO_VERBOSE: bool = True     # Disable console prints (~2-3% speedup)

# Use persistent server mode if available
WHISPER_USE_PERSISTENT: bool = True


# -- TTS (Piper) -----------------------------------------------------------
PIPER_BIN: str = "./piper/piper/piper"
PIPER_VOICE_MODEL: str = "./piper/en_GB-semaine-medium.onnx"
PIPER_SAMPLE_RATE: int = 22050
OUTPUT_DEVICE_NAME: str | int | None = None
TTS_READ_CHUNK_BYTES: int = 32768
TTS_STREAM_BLOCKSIZE: int = 8192
TTS_QUEUE_CHUNK_CHARS: int = 700
TTS_TAIL_SILENCE_S: float = 0.01


# ── Performance Tuning ─────────────────────────────────────
# For Raspberry Pi 5 (4-core ARM Cortex-A76):
# These are recommendations for different targets

class ThreadingProfile:
    """Pre-configured threading profiles for different hardware."""
    
    @staticmethod
    def get_pi_5_profile() -> int:
        """Raspberry Pi 5: 4-core, usually best at 4-6 threads."""
        return 4
    
    @staticmethod
    def get_pi_4_profile() -> int:
        """Raspberry Pi 4: 4-core, often best at 2-4 threads due to thermals."""
        return 2
    
    @staticmethod
    def get_desktop_profile() -> int:
        """Desktop/x86: More cores available, test 8+ threads."""
        return min(os.cpu_count() or 4, 16)
    
    @staticmethod
    def get_profile(hardware: str = "auto") -> int:
        """Get threading profile for target hardware.
        
        Args:
            hardware: "pi5", "pi4", "desktop", or "auto" (detect from cpu_count)
        """
        if hardware == "pi5":
            return ThreadingProfile.get_pi_5_profile()
        elif hardware == "pi4":
            return ThreadingProfile.get_pi_4_profile()
        elif hardware == "desktop":
            return ThreadingProfile.get_desktop_profile()
        else:
            # Auto-detect
            cpu_count = os.cpu_count() or 4
            if cpu_count <= 4:
                return 4  # Conservative for ARM
            else:
                return min(cpu_count, 8)


# ── Latency Profiling ──────────────────────────────────────
# Enable to collect timing data during voice loop

ENABLE_LATENCY_LOGGING: bool = True

class LatencyTimers:
    """Timing labels for profiling the voice loop."""
    RECORD_START = "record_start"
    RECORD_END = "record_end"
    TRANSCRIBE_START = "transcribe_start"
    TRANSCRIBE_END = "transcribe_end"
    CHATBOT_START = "chatbot_start"
    CHATBOT_END = "chatbot_end"
    TTS_START = "tts_start"
    TTS_END = "tts_end"
    CYCLE_END = "cycle_end"
