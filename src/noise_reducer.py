#!/usr/bin/env python3
import numpy as np

class AudioNoiseReducer:
    def __init__(self, sample_rate=16000, frame_size=512, hop_size=256):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = np.hanning(self.frame_size)
        self._noise_profile = None
        self._smoothing = 0.95

    def _estimate_noise_profile(self, samples):
        num_frames = min(8, len(samples) // self.frame_size)
        frames = samples[:num_frames * self.frame_size].reshape(num_frames, self.frame_size)
        mags = np.abs(np.fft.rfft(frames * self.window[:self.frame_size], axis=1))
        return np.min(mags, axis=0)

    def reduce_noise_spectral_subtraction(self, audio):
        if len(audio) < self.frame_size * 2:
            return audio

        samples = np.array(audio, dtype=np.float32)
        num_frames = (len(samples) - self.frame_size) // self.hop_size + 1
        output = np.zeros(len(samples), dtype=np.float32)
        count = np.zeros(len(samples), dtype=np.float32)

        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        voice_mask = (freqs >= 200) & (freqs <= 4000)

        noise_profile = self._estimate_noise_profile(samples)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = samples[start:start + self.frame_size] * self.window
            stft = np.fft.rfft(frame)
            mag = np.abs(stft)
            phase = np.angle(stft)

            # Adaptive noise profile update: track minimum across frames
            noise_profile = np.minimum(
                self._smoothing * noise_profile + (1 - self._smoothing) * mag,
                mag
            )

            # Wiener-style gain: G(k) = SNR(k) / (1 + SNR(k))
            snr = np.maximum(mag / (noise_profile + 1e-8) - 1.0, 0.0)
            gain = snr / (1.0 + snr)

            # Restore speech formant band to prevent voice cutoff
            gain[voice_mask] = np.maximum(gain[voice_mask], 0.92)
            gain = np.clip(gain, 0.10, 1.0)

            clean_stft = mag * gain * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_stft) * self.window
            output[start:start + self.frame_size] += clean_frame
            count[start:start + self.frame_size] += self.window

        mask = count > 1e-8
        output[mask] /= count[mask]

        return output

if __name__ == "__main__":
    reducer = AudioNoiseReducer()
    t = np.linspace(0, 1, 16000)
    signal = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
    noise = np.random.normal(0, 0.25, 16000)
    cleaned = reducer.reduce_noise_spectral_subtraction(signal + noise)
