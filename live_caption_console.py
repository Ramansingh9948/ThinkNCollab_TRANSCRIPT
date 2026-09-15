#!/usr/bin/env python3
"""
ThinkNCollab Real-Time Live Voice Captioning Console
Captures live microphone audio in real-time, runs Wiener Adaptive Noise Filtering,
and displays live scrolling captions using ThinkNCollab PyTorch ASR (206.7M Params).
"""

import os
import sys
import time
import json
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import sounddevice as sd
except ImportError:
    print("Error: 'sounddevice' library is required for live mic capture.")
    print("Please install it using: pip install sounddevice")
    sys.exit(1)

try:
    from thinkncollab_whisper import load_model
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from thinkncollab_whisper import load_model

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 2.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)


class LiveVoiceCaptioner:
    def __init__(self, language="hindi", model_size="small"):
        self.language = language
        print("=" * 70)
        print("   ThinkNCollab Real-Time Live Voice Transcriber & Captioner")
        print("   Engine: ThinkNCollab PyTorch Model (206,726,656 Parameters)")
        print(f"   Language Mode: {language.upper()}")
        print("=" * 70)
        print("\nLoading PyTorch ASR Engine...")
        self.asr_engine = load_model(name=model_size)
        print("ASR Engine ready!\n")
        self.transcript_history = []

    def audio_callback(self, indata, frames, time_info, status):
        pass

    def start_live_captioning(self):
        print(f"🎙️  LISTENING ON MICROPHONE... (Speak now in {self.language.upper()})")
        print("Press Ctrl+C to stop live captioning.\n")
        print("-" * 70)

        chunk_counter = 0

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
                while True:
                    audio_chunk, overflowed = stream.read(CHUNK_SAMPLES)
                    chunk_counter += 1
                    t0 = time.time()

                    audio_data = audio_chunk.flatten()

                    # RMS Volume Energy Check — filter out background room silence
                    rms_energy = np.sqrt(np.mean(audio_data ** 2))
                    if rms_energy < 0.015:
                        continue

                    # Save temporary wave file for inference
                    import tempfile, soundfile as sf
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        sf.write(tmp_path, audio_data, SAMPLE_RATE)

                    result = self.asr_engine.transcribe(tmp_path, language=self.language)

                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                    text = result.get("raw_text", "").strip()
                    text = text.replace("▁", " ").replace("\u2581", " ").strip()
                    import re
                    text = re.sub(r"\s+", " ", text)

                    infer_ms = result.get("infer_ms", 0.0)
                    timestamp = time.strftime("%H:%M:%S")

                    if text and not re.match(r"^[\s\W▁]+$", text):
                        line = f"[{timestamp}] Speaker 1: {text}"
                        print(f"  {line}  ({infer_ms}ms)")
                        self.transcript_history.append(line)

        except KeyboardInterrupt:
            print("\n" + "-" * 70)
            print("Stopped Live Voice Captioning.")
            self.save_log()

    def save_log(self, filename="live_captions_log.txt"):
        if self.transcript_history:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"--- ThinkNCollab Live Caption Session ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")
                for line in self.transcript_history:
                    f.write(line + "\n")
            print(f"Live caption transcript log saved to '{filename}'.")


def main():
    parser = argparse.ArgumentParser(description="ThinkNCollab Real-Time Live Voice Captioning Console")
    parser.add_argument("--language", type=str, default="hindi",
                        choices=["hindi", "hinglish", "english", "bengali_hindi", "tamil_hindi", "rajasthani_hindi"],
                        help="Language mode")
    parser.add_argument("--model", type=str, default="small", help="Model size: 'small'")

    args = parser.parse_args()

    captioner = LiveVoiceCaptioner(language=args.language, model_size=args.model)
    captioner.start_live_captioning()


if __name__ == "__main__":
    main()
