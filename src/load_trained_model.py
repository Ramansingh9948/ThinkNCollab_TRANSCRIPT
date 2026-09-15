#!/usr/bin/env python3
import os
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints"
)

class WhisperSmallHinglish(nn.Module):
    def __init__(self, vocab_size=4096, n_mels=80):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, 768, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(768, 768, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.0
            ),
            num_layers=12
        )
        self.embedding = nn.Embedding(vocab_size, 768)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.0
            ),
            num_layers=12
        )
        self.head = nn.Linear(768, vocab_size)

    def forward(self, mel_input, decoder_input):
        x = self.stem(mel_input)
        x = x.permute(0, 2, 1)
        memory = self.encoder(x)
        tgt = self.embedding(decoder_input)
        out = self.decoder(tgt, memory)
        return self.head(out)


def load_local_trained_model():
    v2_path = os.path.join(CHECKPOINT_DIR, "whisper_small_hinglish_v2.pt")
    pt_path = os.path.join(CHECKPOINT_DIR, "whisper_small_hinglish_final.pt")
    dir_path = os.path.join(CHECKPOINT_DIR, "whisper_small_hinglish_final")

    checkpoint_path = None
    if os.path.isfile(v2_path):
        checkpoint_path = v2_path
    elif os.path.isfile(pt_path):
        checkpoint_path = pt_path
    elif os.path.isdir(dir_path):
        checkpoint_path = dir_path
    
    if not checkpoint_path:
        return None

    model = WhisperSmallHinglish(vocab_size=4096, n_mels=80)

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        elif isinstance(state, nn.Module):
            model = state
    except Exception:
        return None

    model.eval()
    return model


if __name__ == "__main__":
    model = load_local_trained_model()
    if model:
        total = sum(p.numel() for p in model.parameters())
        print(f"ThinkNCollab model loaded: {total:,} parameters")
        for name, child in model.named_children():
            params = sum(p.numel() for p in child.parameters())
            print(f"  {name:12s}: {type(child).__name__:30s}  ({params:,} params)")
    else:
        print("Model not found.")
