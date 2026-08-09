#!/usr/bin/env python3
import math
import numpy as np

class AudioNoiseReducer:
    def __init__(self, sample_rate=16000, alpha=2.0, beta=0.01):
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.beta = beta

    def reduce_noise_spectral_subtraction(self, audio_samples):
        if len(audio_samples) == 0:
            return audio_samples

        samples = np.array(audio_samples, dtype=np.float32)
        frame_len = 512
        hop_len = 256

        if len(samples) < frame_len * 5:
            return audio_samples

        noise_frames = samples[:frame_len * 5]
        noise_stft = np.abs(np.fft.rfft(np.reshape(noise_frames, (5, frame_len)), axis=1))
        noise_psd = np.mean(noise_stft ** 2, axis=0)

        num_frames = (len(samples) - frame_len) // hop_len + 1
        cleaned_signal = np.zeros(len(samples), dtype=np.float32)
        window = np.hanning(frame_len)

        for i in range(num_frames):
            start = i * hop_len
            frame = samples[start:start + frame_len] * window
            stft = np.fft.rfft(frame)
            mag = np.abs(stft)
            phase = np.angle(stft)
            psd = mag ** 2

            subtracted_psd = np.maximum(psd - self.alpha * noise_psd, self.beta * psd)
            clean_mag = np.sqrt(subtracted_psd)

            clean_stft = clean_mag * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_stft) * window

            cleaned_signal[start:start + frame_len] += clean_frame

        return cleaned_signal

if __name__ == "__main__":
    reducer = AudioNoiseReducer()
    t = np.linspace(0, 1, 16000)
    speech = np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.3, 16000)
    noisy_audio = speech + noise

    clean = reducer.reduce_noise_spectral_subtraction(noisy_audio)
