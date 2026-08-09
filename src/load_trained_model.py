#!/usr/bin/env python3
import os
import sys
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

POSSIBLE_PATHS = [
    "checkpoints/whisper_small_hinglish_final.pt",
    "checkpoints/whisper_small_hinglish_final"
]

def load_local_trained_model():
    target_path = None
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            target_path = p
            break

    if not target_path:
        return None

    try:
        try:
            from src.whisper_model import PyTorchWhisperSmallModel
        except ModuleNotFoundError:
            from whisper_model import PyTorchWhisperSmallModel

        state_dict = None
        try:
            state_dict = torch.load(target_path, map_location="cpu", weights_only=False)
        except Exception:
            state_dict = torch.load(target_path, map_location="cpu")

        model = PyTorchWhisperSmallModel()
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    except Exception:
        return True

if __name__ == "__main__":
    model = load_local_trained_model()
    print("Loaded model instance:", model)
