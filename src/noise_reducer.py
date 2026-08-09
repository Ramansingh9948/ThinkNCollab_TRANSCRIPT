#!/usr/bin/env python3
import numpy as np

class AudioNoiseReducer:
    def __init__(self, sample_rate=16000, frame_size=512, hop_size=256):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = np.hanning(self.frame_size)

    def reduce_noise_spectral_subtraction(self, audio):
        if len(audio) < self.frame_size * 2:
            return audio

        samples = np.array(audio, dtype=np.float32)
        num_frames = (len(samples) - self.frame_size) // self.hop_size + 1
        output = np.zeros(len(samples), dtype=np.float32)

        # Estimate initial noise spectrum profile from ambient background
        noise_frames = samples[:self.frame_size * 4]
        noise_stft = np.abs(np.fft.rfft(np.reshape(noise_frames, (4, self.frame_size)), axis=1))
        noise_profile = np.median(noise_stft, axis=0)

        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        voice_mask = (freqs >= 250) & (freqs <= 3800)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = samples[start:start + self.frame_size] * self.window
            stft = np.fft.rfft(frame)
            mag = np.abs(stft)
            phase = np.angle(stft)

            # Adaptive SNR-based Gain Masking
            snr = np.maximum(mag / (noise_profile + 1e-8), 1.0)
            gain = 1.0 - (1.0 / snr)
            gain[voice_mask] = np.maximum(gain[voice_mask], 0.85)
            gain = np.clip(gain, 0.15, 1.0)

            clean_stft = mag * gain * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_stft) * self.window
            output[start:start + self.frame_size] += clean_frame

        return output

if __name__ == "__main__":
    reducer = AudioNoiseReducer()
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) + np.random.normal(0, 0.2, 16000)
    cleaned = reducer.reduce_noise_spectral_subtraction(signal)
