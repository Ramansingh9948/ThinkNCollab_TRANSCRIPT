#!/usr/bin/env python3
import os
import sys
import argparse
import time
import json
import struct
import wave
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    from src.noise_reducer import AudioNoiseReducer
    from src.load_trained_model import load_local_trained_model
except (ModuleNotFoundError, ImportError):
    try:
        from noise_reducer import AudioNoiseReducer
        from load_trained_model import load_local_trained_model
    except (ModuleNotFoundError, ImportError):
        class AudioNoiseReducer:
            def __init__(self, sample_rate=16000): pass
            def reduce_noise_spectral_subtraction(self, audio): return audio
        def load_local_trained_model(): return None

# Load SentencePiece Tokenizer or BPE JSON
SPM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tnc_tokenizer.model")
BPE_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hinglish_bpe.json")

SP_PROCESSOR = None
ID_TO_TOKEN = {}

if os.path.exists(SPM_MODEL_PATH):
    try:
        import sentencepiece as spm
        SP_PROCESSOR = spm.SentencePieceProcessor()
        SP_PROCESSOR.load(SPM_MODEL_PATH)
    except Exception:
        pass

if os.path.exists(BPE_JSON_PATH):
    try:
        with open(BPE_JSON_PATH, "r", encoding="utf-8") as f:
            bpe_data = json.load(f)
            ID_TO_TOKEN = {idx: tok for tok, idx in bpe_data.get("tokens", {}).items()}
    except Exception:
        pass

LANGUAGE_ALLOWED_TOKENS = {}

def get_allowed_tokens_for_language(lang):
    if lang in LANGUAGE_ALLOWED_TOKENS:
        return LANGUAGE_ALLOWED_TOKENS[lang]

    import unicodedata
    allowed = {0, 1, 2, 3, 500}  # pad, bos, eos, unk, space piece

    if SP_PROCESSOR is not None:
        for i in range(260, SP_PROCESSOR.get_piece_size()):
            s = SP_PROCESSOR.decode([i])
            if lang in ["hindi", "rajasthani_hindi"]:
                if any("DEVANAGARI" in unicodedata.name(ch, "") for ch in s):
                    allowed.add(i)
            elif lang in ["hinglish", "english"]:
                if any("LATIN" in unicodedata.name(ch, "") or ch.isascii() for ch in s):
                    allowed.add(i)
            elif lang == "bengali_hindi":
                if any("BENGALI" in unicodedata.name(ch, "") for ch in s):
                    allowed.add(i)
            elif lang == "tamil_hindi":
                if any("TAMIL" in unicodedata.name(ch, "") for ch in s):
                    allowed.add(i)

    LANGUAGE_ALLOWED_TOKENS[lang] = allowed
    return allowed

class ThinkNCollabWhisperModel:
    def __init__(self, model_name="small", device="auto"):
        self.model_name = model_name
        import torch
        if device == "auto" or device == "cpu":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.noise_reducer = AudioNoiseReducer(sample_rate=16000)
        self.model = load_local_trained_model()
        if self.model is not None:
            self.model = self.model.to(self.device)

    def _audio_to_log_mel(self, audio_path, sample_rate=16000, n_mels=80, max_frames=800):
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            y_clean = self.noise_reducer.reduce_noise_spectral_subtraction(y)
            # Check if input audio is silent
            if np.max(np.abs(y_clean)) < 0.015:
                return np.zeros((n_mels, max_frames), dtype=np.float32)

            # Cap max length to 8s (128,000 samples)
            if len(y_clean) > 8 * 16000:
                y_clean = y_clean[:8 * 16000]

            mel = librosa.feature.melspectrogram(
                y=np.array(y_clean, dtype=np.float32),
                sr=sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=n_mels
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
            if log_mel.shape[1] < max_frames:
                log_mel = np.pad(log_mel, ((0,0),(0, max_frames - log_mel.shape[1])))
            else:
                log_mel = log_mel[:, :max_frames]
            return log_mel.astype(np.float32)
        except Exception:
            return np.zeros((n_mels, max_frames), dtype=np.float32)

    def transcribe(self, audio_input, language="hindi", task="transcribe", temperature=0.0, verbose=False):
        t0 = time.perf_counter()
        file_name = os.path.basename(audio_input) if isinstance(audio_input, str) else "audio_input.wav"

        if verbose:
            print(f"Ingesting '{file_name}' (language={language}, task={task})")

        log_mel = self._audio_to_log_mel(audio_input) if isinstance(audio_input, str) and os.path.exists(audio_input) else np.zeros((80, 800), dtype=np.float32)

        # Silence Gate: if audio contains only silence or near-zero energy, return empty
        if np.max(log_mel) < 0.05 or np.ptp(log_mel) < 0.02:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "text": "",
                "raw_text": "",
                "segments": [],
                "language": language,
                "task": task,
                "infer_ms": round(elapsed_ms, 1)
            }

        import torch
        generated_tokens = [1]  # <s> start token
        allowed_tokens = get_allowed_tokens_for_language(language)

        if self.model is not None:
            mel_tensor = torch.tensor(log_mel, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                for i in range(15):
                    dec_in = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)
                    logits = self.model(mel_tensor, dec_in)
                    step_logits = logits[0, -1, :]

                    if allowed_tokens:
                        mask = torch.full_like(step_logits, -float("inf"))
                        for tid in allowed_tokens:
                            if tid < len(step_logits):
                                mask[tid] = step_logits[tid]
                        next_tok = int(mask.argmax(dim=-1).item())
                    else:
                        next_tok = int(step_logits.argmax(dim=-1).item())

                    if len(generated_tokens) > 2 and next_tok == generated_tokens[-1] == generated_tokens[-2]:
                        break
                    generated_tokens.append(next_tok)
                    if next_tok == 2:  # </s> EOS
                        break

        # Decode tokens to clean text using SentencePiece or BPE
        text_out = ""
        valid_tokens = [t for t in generated_tokens if t not in (0, 1, 2, 3)]

        if SP_PROCESSOR is not None and valid_tokens:
            try:
                text_out = SP_PROCESSOR.decode(valid_tokens)
            except Exception:
                text_out = ""

        if not text_out and valid_tokens:
            words = [ID_TO_TOKEN.get(t, "") for t in valid_tokens if ID_TO_TOKEN.get(t, "")]
            text_out = " ".join([w for w in words if w and not w.startswith("<")])

        # Clean SentencePiece space symbols and whitespace
        if text_out:
            text_out = text_out.replace("\u2581", " ").replace("▁", " ").strip()
            import re
            text_out = re.sub(r"\s+", " ", text_out)

        # If only non-word punctuation/symbols were decoded, treat as empty
        import re
        if text_out and re.match(r"^[\s\W▁]+$", text_out):
            text_out = ""

        elapsed_ms = (time.perf_counter() - t0) * 1000
        timestamp = time.strftime("%M:%S")

        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "speaker": "Speaker 1",
                "text": text_out
            }
        ]

        return {
            "text": f"[{timestamp}] Speaker 1: {text_out}",
            "raw_text": text_out,
            "segments": segments,
            "language": language,
            "task": task,
            "infer_ms": round(elapsed_ms, 1)
        }

def load_model(name="small", device="cpu"):
    return ThinkNCollabWhisperModel(model_name=name, device=device)

def main():
    parser = argparse.ArgumentParser(
        description="ThinkNCollab-Whisper: High-Efficiency Open-Source Speech-to-Text Transcriber"
    )
    parser.add_argument("audio", type=str, help="Path to input audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--model", type=str, default="small", help="Model size: 'small'")
    parser.add_argument("--language", type=str, default="hindi",
                        choices=["hindi", "hinglish", "english", "bengali_hindi", "tamil_hindi", "rajasthani_hindi"],
                        help="Language mode")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task mode")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature")
    parser.add_argument("--output_format", type=str, default="txt", choices=["txt", "json"], help="Output format")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' not found.")
        sys.exit(1)

    model = load_model(name=args.model)
    result = model.transcribe(args.audio, language=args.language, task=args.task, temperature=args.temperature, verbose=args.verbose)

    print(result["text"])

    base_name = os.path.splitext(os.path.basename(args.audio))[0]
    out_path = os.path.join(args.output_dir, f"{base_name}_transcript.{args.output_format}")

    with open(out_path, "w", encoding="utf-8") as f:
        if args.output_format == "json":
            json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            f.write(result["text"] + "\n")

if __name__ == "__main__":
    main()
