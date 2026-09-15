#!/usr/bin/env python3
"""
ThinkNCollab Multi-Language ASR — Local Mac M5 Training Script
Supports: Hindi, Hinglish, Indian English, Bengali-Hindi, Tamil-Hindi, Rajasthani-Hindi
Device: Apple M5 MPS GPU (Metal Performance Shaders)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]    = "TRUE"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_HOME"]                 = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
os.environ["HF_DATASETS_CACHE"]       = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache", "datasets")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device setup — M5 MPS GPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple M5 MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (no GPU found)")

# ── Hyperparameters ──────────────────────────────────────────────────────────
VOCAB_SIZE       = 4096
N_MELS           = 80
D_MODEL          = 768
N_HEADS          = 12
ENC_LAYERS       = 12
DEC_LAYERS       = 12
FF_DIM           = 3072
MAX_FRAMES       = 3000   # 30 seconds @ 16kHz / hop=160
MAX_TARGET_LEN   = 128
BATCH_SIZE       = 2      # Small batch for 16GB RAM
GRAD_ACCUM       = 8      # Effective batch = 16
LEARNING_RATE    = 3e-4
MAX_EPOCHS       = 5
MAX_SAMPLES      = 5000   # Start small — increase when confident

# ── Model Architecture ───────────────────────────────────────────────────────
class WhisperSmallHinglish(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_MELS, D_MODEL, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(D_MODEL, D_MODEL, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
                dropout=0.1, batch_first=True, norm_first=True
            ),
            num_layers=ENC_LAYERS
        )
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
                dropout=0.1, batch_first=True, norm_first=True
            ),
            num_layers=DEC_LAYERS
        )
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.embedding.weight

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, mel):
        x = self.stem(mel).permute(0, 2, 1)
        return self.encoder(x)

    def forward(self, mel, tokens):
        memory = self.encode(mel)
        T = tokens.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=mel.device)
        tgt = self.embedding(tokens)
        out = self.decoder(tgt, memory, tgt_mask=mask)
        return self.head(out)


# ── Audio Feature Extraction ─────────────────────────────────────────────────
import librosa

def audio_to_log_mel(waveform, sr=16000):
    y = np.array(waveform, dtype=np.float32)
    if y.max() > 1.0:
        y = y / 32768.0
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
    if log_mel.shape[1] < MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0,0),(0, MAX_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    return log_mel.astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────────────
class TNCDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples   = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio, text = self.samples[idx]
        try:
            if isinstance(audio, dict):
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio.get("sampling_rate", 16000)
                if sr != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            else:
                waveform, _ = librosa.load(str(audio), sr=16000, mono=True)

            mel = audio_to_log_mel(waveform)
        except Exception:
            mel = np.zeros((N_MELS, MAX_FRAMES), dtype=np.float32)

        mel_tensor = torch.tensor(mel, dtype=torch.float32)

        token_ids = [1] + self.tokenizer.encode(text)[:MAX_TARGET_LEN-2] + [2]
        token_ids += [0] * (MAX_TARGET_LEN - len(token_ids))
        token_tensor = torch.tensor(token_ids[:MAX_TARGET_LEN], dtype=torch.long)

        return mel_tensor, token_tensor


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_datasets(max_samples_per_ds=1000):
    from datasets import load_dataset
    samples = []

    sources = [
        ("ai4bharat/kathbath",                     "hindi",     "sentence"),
        ("mozilla-foundation/common_voice_11_0",   "hi",        "sentence"),
        ("google/fleurs",                           "hi_in",     "transcription"),
        ("google/fleurs",                           "en_in",     "transcription"),
        ("mozilla-foundation/common_voice_11_0",   "bn",        "sentence"),
        ("google/fleurs",                           "bn_in",     "transcription"),
        ("mozilla-foundation/common_voice_11_0",   "ta",        "sentence"),
        ("google/fleurs",                           "ta_in",     "transcription"),
    ]

    for name, config, text_key in sources:
        try:
            ds = load_dataset(name, config, split="train", streaming=True)
            count = 0
            for item in ds:
                audio = item.get("audio")
                text  = item.get(text_key, item.get("sentence", item.get("text", "")))
                if audio and text and text.strip():
                    samples.append((audio, text.strip()))
                    count += 1
                    if count >= max_samples_per_ds:
                        break
            print(f"  [{name} / {config}] {count} samples")
        except Exception as e:
            print(f"  [SKIP] {name}/{config}: {e}")

    print(f"Total samples: {len(samples)}")
    return samples


# ── Tokenizer ────────────────────────────────────────────────────────────────
def train_or_load_tokenizer(samples):
    import sentencepiece as spm

    vocab_json_path  = "data/hinglish_bpe.json"
    spm_model_path   = "data/tnc_tokenizer.model"
    corpus_path      = "data/tnc_corpus_local.txt"

    if os.path.exists(spm_model_path):
        print(f"Loading existing tokenizer from '{spm_model_path}'")
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_path)
        return sp

    print("Training SentencePiece tokenizer...")
    os.makedirs("data", exist_ok=True)

    # Write corpus from samples
    lines = [text for _, text in samples if text.strip()]

    # Fallback: if no real samples loaded, use a built-in seed corpus
    if len(lines) < 50:
        print("  Using built-in seed corpus (no external data loaded)")
        lines += [
            "hello am i audible", "आज की मीटिंग शुरू हो चुकी है",
            "Aaj ki meeting start ho chuki hai", "meeting start karo bhai",
            "please share your screen", "can you hear me clearly",
            "আজকের মিটিং শুরু হয়েছে", "இன்றைய கூட்டம் தொடங்கிவிட்டது",
            "आज री बैठक शुरू हो ग्यी है", "project update de do yaar",
            "kal meeting hai office mein", "आप सभी का स्वागत है",
            "voice clear aa rahi hai kya", "sound theek hai na",
            "today ki meeting 3 baje hai", "presentation ready kar lo",
            "background noise bahut hai", "thoda aur loud bolo",
            "kya aap mujhe sun sakte hain", "हां बिल्कुल सुनाई दे रहा है",
            "status update kya hai project ka", "deadline kab hai",
            "code review complete ho gaya", "deployment kab karein",
            "client call aaj 5 baje hai", "meeting reschedule karni padegi",
            "audio quality bahut kharab hai", "thoda mic se door ho jao",
            "notes likh raha hoon abhi", "presentation mein kya add karein",
            "kal tak report ready karni hai", "team ko brief kar do",
            "Tamil mein bolne ka try kar", "Bengali mein conference hai",
            "Rajasthani accent mein training data chahiye",
            "south Indian English accent difficult hai",
        ] * 20  # Repeat to get enough data

    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"  Corpus size: {len(lines)} lines")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix="data/tnc_tokenizer",
        vocab_size=min(VOCAB_SIZE, len(set("".join(lines))) + 500),
        character_coverage=0.9998,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        byte_fallback=True,
        split_digits=True,
        add_dummy_prefix=False
    )

    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)

    vocab_map = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(vocab_map), "special_tokens": ["<pad>","<s>","</s>","<unk>"], "tokens": vocab_map}, f, ensure_ascii=False, indent=2)

    print(f"Tokenizer trained: {sp.get_piece_size()} tokens saved to '{spm_model_path}'")
    return sp


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(model, loader, epochs=MAX_EPOCHS):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (mel_batch, token_batch) in enumerate(pbar):
            mel_batch   = mel_batch.to(DEVICE)
            token_batch = token_batch.to(DEVICE)

            dec_input  = token_batch[:, :-1]
            dec_target = token_batch[:, 1:]

            logits = model(mel_batch, dec_input)
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), dec_target.reshape(-1))
            loss   = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} done. Loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "loss": avg_loss,
            "vocab_size": VOCAB_SIZE
        }, f"checkpoints/tnc_epoch_{epoch+1}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/whisper_small_hinglish_v2.pt")
            print(f"  Best model saved (loss: {best_loss:.4f})")

    print("Training complete!")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ThinkNCollab Multi-Language ASR Trainer")
    parser.add_argument("--samples",  type=int, default=MAX_SAMPLES, help="Max samples per dataset")
    parser.add_argument("--epochs",   type=int, default=MAX_EPOCHS,  help="Training epochs")
    parser.add_argument("--batch",    type=int, default=BATCH_SIZE,  help="Batch size")
    parser.add_argument("--resume",   type=str, default=None,        help="Resume from checkpoint path")
    args = parser.parse_args()

    print(f"Device       : {DEVICE}")
    print(f"Max Samples  : {args.samples} per dataset")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch}  (effective: {args.batch * GRAD_ACCUM})")

    # 1. Load data
    print("\nLoading datasets...")
    samples = load_datasets(max_samples_per_ds=args.samples)

    # 2. Train tokenizer
    print("\nPreparing tokenizer...")
    tokenizer = train_or_load_tokenizer(samples)

    # 3. Build dataset
    dataset = TNCDataset(samples, tokenizer)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    print(f"Dataset ready: {len(dataset)} samples")

    # 4. Build model
    model = WhisperSmallHinglish().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model ready: {total:,} parameters on {DEVICE}")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Resumed from checkpoint: {args.resume}")

    # 5. Train
    print("\nStarting training...")
    train(model, loader, epochs=args.epochs)

if __name__ == "__main__":
    main()
