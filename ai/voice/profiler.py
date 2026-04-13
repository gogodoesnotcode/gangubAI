#!/usr/bin/env python3
"""Latency profiling tool for voice pipeline.

Usage:
    # Profile record -> transcribe latency only
    python3 -m ai.voice.profiler --profile-stt gangubai_input.wav
    
    # Benchmark thread counts on your Pi
    python3 -m ai.voice.profiler --benchmark-threads gangubai_input.wav
    
    # Silence timeout tuning
    python3 -m ai.voice.profiler --test-silence-timeout
    
    # CPU governor check
    python3 -m ai.voice.profiler --check-cpu-governor
"""

from __future__ import annotations

import argparse
import subprocess
import time
import sys
from pathlib import Path

from ai.voice.transcriber_persistent import transcribe_fast, benchmark_thread_count
from ai.voice import config


def check_cpu_governor() -> None:
    """Check CPU frequency scaling on Linux systems."""
    print("[CPU Governor Check]")
    try:
        result = subprocess.run(
            ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
            capture_output=True,
            text=True,
            check=False,
        )
        governor = result.stdout.strip()
        print(f"  Current: {governor}")
        
        if governor != "performance":
            print(f"\n  ⚠️  CPU is in '{governor}' mode")
            print(f"  For benchmarking, consider setting to 'performance':")
            print(f"    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            print(f"\n  (This temporarily boosts CPU to max frequency)")
    except FileNotFoundError:
        print("  [Not Linux or sysfs not accessible - skipping]")
    except Exception as e:
        print(f"  Error: {e}")


def profile_stt(wav_path: str, runs: int = 5) -> None:
    """Profile STT latency in isolation (record -> transcribe).
    
    This excludes chatbot and TTS timing to measure pure transcription speed.
    """
    print(f"[STT Latency Profile] ({runs} runs)")
    print(f"  Model: {Path(config.WHISPER_CPP_MODEL).name}")
    print(f"  Threads: {config.WHISPER_CPP_THREADS}")
    print(f"  Flags: no-timestamps={config.WHISPER_NO_TIMESTAMPS}, no-verbose={config.WHISPER_NO_VERBOSE}")
    print()

    times = []
    for i in range(runs):
        try:
            text, elapsed = transcribe_fast(
                wav_path,
                threads=config.WHISPER_CPP_THREADS,
                return_timing=True,
            )
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s - {len(text)} chars")
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {e}")
            return

    print()
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"  Average: {avg:.3f}s")
    print(f"  Min: {min_t:.3f}s")
    print(f"  Max: {max_t:.3f}s")
    print(f"  Range: {max_t - min_t:.3f}s")

    # Guidance
    if avg < 1.0:
        print(f"\n  ✓ Excellent (< 1.0s per turn)")
    elif avg < 2.0:
        print(f"\n  ✓ Good (1-2s per turn)")
    elif avg < 3.0:
        print(f"\n  ⚠️  Fair (2-3s per turn)")
    else:
        print(f"\n  ⚠️  Slow (> 3s per turn)")


def compare_thread_settings(wav_path: str) -> None:
    """Compare different thread counts to find optimal for your hardware."""
    print("[Thread Count Benchmark]")
    print(f"  Testing: 2, 4, 6, 8 threads")
    print(f"  (Run 3x each to get reliable average)\n")
    
    results = benchmark_thread_count(wav_path, runs_per_config=3)
    
    # Find best
    best_threads = min(results, key=lambda t: sum(results[t]) / len(results[t]))
    best_avg = sum(results[best_threads]) / len(results[best_threads])
    
    print(f"\n  ✓ Recommended: {best_threads} threads ({best_avg:.3f}s average)")
    print(f"    Update ai/voice/config.py: WHISPER_CPP_THREADS = {best_threads}")


def test_silence_timeout(duration_s: float = 5.0) -> None:
    """Simulate voice recording with different silence timeouts.
    
    Shows how SILENCE_DURATION_S affects turn cutoff time.
    """
    print(f"[Silence Timeout Test] ({duration_s}s simulation)")
    print(f"  Current setting: {config.SILENCE_DURATION_S}s")
    print()

    for timeout in [0.5, 0.7, 1.0, 1.5, 2.0]:
        print(f"  If SILENCE_DURATION_S = {timeout}s:")
        total = config.PRE_RECORD_DELAY_S + timeout
        print(f"    Total latency added: ~{total:.1f}s")
        print(f"    (0.4s pre-delay + {timeout}s silence wait)")

    print(f"\n  Current total: ~{config.PRE_RECORD_DELAY_S + config.SILENCE_DURATION_S:.1f}s")
    print(f"\n  Recommendations:")
    print(f"    - Fast response (0.7s silence): Good for clear audio, may cut off slow speech")
    print(f"    - Medium (1.0s silence): Balanced, current optimized setting")
    print(f"    - Defensive (1.5s silence): Good for background noise, slower response")


def profile_full_cycle(record_file: str | None = None) -> None:
    """Profile full cycle: record -> transcribe -> dummy chat -> TTS."""
    print("[Full Cycle Profile]")
    print("  (This requires audio input - run with --wav-input for testing)")
    print()
    print("  To profile real conditions:")
    print("    python3 -m ai.voice.app --no-tts --wav-input --wav-file <your_audio.wav>")
    print()
    print("  Look for timing printouts in output (if latency logging enabled)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Latency profiling for GangubAI voice pipeline"
    )
    parser.add_argument("wav_path", nargs="?", help="Path to test audio file")
    parser.add_argument(
        "--profile-stt",
        action="store_true",
        help="Profile STT latency (5 runs)",
    )
    parser.add_argument(
        "--benchmark-threads",
        action="store_true",
        help="Benchmark thread counts (2,4,6,8)",
    )
    parser.add_argument(
        "--check-cpu",
        action="store_true",
        help="Check CPU governor settings",
    )
    parser.add_argument(
        "--test-silence",
        action="store_true",
        help="Show silence timeout impact",
    )
    parser.add_argument(
        "--profile-full",
        action="store_true",
        help="Profile full voice loop",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks (requires audio file)",
    )

    args = parser.parse_args()

    if args.check_cpu or args.all:
        check_cpu_governor()
        if not (args.all and args.wav_path):
            print()

    if args.test_silence or args.all:
        test_silence_timeout()
        if not (args.all and args.wav_path):
            print()

    if args.profile_stt or (args.all and args.wav_path):
        if not args.wav_path:
            print("[ERROR] --profile-stt requires a WAV file path")
            return 1
        profile_stt(args.wav_path)
        if not (args.all and args.benchmark_threads):
            print()

    if args.benchmark_threads or (args.all and args.wav_path):
        if not args.wav_path:
            print("[ERROR] --benchmark-threads requires a WAV file path")
            return 1
        print()
        compare_thread_settings(args.wav_path)
        print()

    if args.profile_full or (args.all and args.wav_path):
        profile_full_cycle(args.wav_path)

    if not any([args.profile_stt, args.benchmark_threads, args.check_cpu, 
                args.test_silence, args.profile_full, args.all]):
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
