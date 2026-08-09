# ThinkNCollab-Whisper: Open-Source Hinglish Speech-to-Text (ASR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Model Size](https://img.shields.io/badge/model--size-%3C1GB-green.svg)]()
[![Version](https://img.shields.io/badge/version-1.0.1-brightgreen.svg)]()
[![Languages](https://img.shields.io/badge/languages-Hindi%20%7C%20Indian--English%20%7C%20Hinglish-orange.svg)]()

> **Scratch-Trained Small (<1 GB) ASR Model for Hindi + Indian-English (Hinglish) Code-Switching.**  
> Trained on 1,000+ hours of multi-dataset speech corpora (AI4Bharat Kathbath, Mozilla Common Voice, Google FLEURS, MUCS). Auto-binds to any server IP on any available port, providing a 100% ThinkNCollab Native REST API.

---

## Key Features

- **Open-Source & 100% Free**: No API keys required, 100% offline local inference capability.
- **Hinglish Code-Switching Support**: High accuracy for Hindi (Devanagari/Roman) + Indian English mixed speech.
- **Multi-Dataset Training Scaling**: 1,000+ hours of training data across regional Indian accents.
- **Automatic Server Binding (0.0.0.0) & Auto-Port Detection**: Runs on any Linux, Mac, or Windows server.
- **ThinkNCollab Native REST API Specification**: Supports `/v1/audio/transcriptions` and `/v1/audio/translations`.
- **Spectral Subtraction Noise Gate**: Automatic background noise reduction for fan, AC hum, and street noise.
- **Official ThinkNCollab JavaScript SDK (@thinkncollab/scribe) & Python Package (thinkncollab_whisper)**.

---

## 1. Direct GitHub Installation

Install ThinkNCollab-Whisper directly into any Python project or environment:

```bash
pip install git+https://github.com/Ramansingh9948/ThinkNCollab_TRANSCRIPT.git
```

### Usage in Python:

```python
import thinkncollab_whisper

# Load ThinkNCollab Whisper Small Model
model = thinkncollab_whisper.load_model("small")

# Transcribe Speech Audio
result = model.transcribe("meeting_audio.wav", language="hinglish", task="transcribe")
print(result["text"])
```

---

## 2. Quickstart Server Execution

Run the ASR server on ANY host machine or cloud server (AWS, GCP, DigitalOcean, Mac):

```bash
# Clone Repository & Install Dependencies
git clone https://github.com/Ramansingh9948/ThinkNCollab_TRANSCRIPT.git
cd ThinkNCollab_TRANSCRIPT

pip install -r requirements.txt

# Start Server (Auto-binds open port on 0.0.0.0)
python3 server.py
```

### Server Output Example:

```text
==========================================================================
  ThinkNCollab ASR API Server Active on Host: 0.0.0.0
==========================================================================
  - Server IP / Host   : http://192.168.1.15:8000
  - ThinkNCollab Endpoint: http://192.168.1.15:8000/v1/audio/transcriptions
  - Network Binding    : 0.0.0.0 (Accessible across server network/internet)
==========================================================================
```

---

## 3. Official ThinkNCollab JavaScript SDK (@thinkncollab/scribe)

Install and use ThinkNCollab's native JavaScript/TypeScript SDK:

```bash
npm install @thinkncollab/scribe
```

```javascript
const { ThinkNCollabScribe } = require('@thinkncollab/scribe');

// Initialize ThinkNCollab Client with Server Host
const scribe = new ThinkNCollabScribe({
    serverUrl: 'http://<YOUR_SERVER_IP>:8000',
    language: 'hinglish'
});

async function run() {
    const result = await scribe.transcribe('meeting_audio.mp3');
    console.log("Transcript:", result.text);
}

run();
```

---

## 4. Terminal CLI Usage

Use ThinkNCollab-Whisper from command line:

```bash
# Terminal CLI Command
python3 thinkncollab_whisper.py meeting.wav --language hinglish --output_format txt
```

---

## Project Directory Layout

```text
ThinkNCollab_TRANSCRIPT/
├── README.md                  # Production Open-Source Documentation
├── LICENSE                    # MIT License Specification
├── CONTRIBUTING.md            # Open-Source Contribution Guidelines
├── server.py                  # Auto-Port REST API Server (0.0.0.0 Host)
├── thinkncollab_whisper.py    # CLI & Python Package Script Module
├── thinkncollab-scribe-sdk.js # Official ThinkNCollab JavaScript / TypeScript SDK Client
├── package.json               # NPM Package Manifest (@thinkncollab/scribe v1.0.1)
├── setup.py                   # PyPI Package Manifest (thinkncollab-scribe v1.0.1)
├── requirements.txt           # Core Python Dependencies
├── checkpoints/               # Trained Model Weights (whisper_small_hinglish_final)
├── src/                       # Core Engine Modules (Noise Reducer, Data Selector)
└── notebooks/                 # Kaggle Multi-Dataset CUDA Training Pipeline
```

---

## Contributing

We welcome community contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on submitting pull requests and reporting issues.

---

## License

MIT License &copy; 2026 ThinkNCollab AI Team. Open-Source for the community.
