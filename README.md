# ThinkNCollab-Whisper: Open-Source Hinglish Speech-to-Text (ASR)

> **Scratch-Trained Small (<1 GB) ASR Model for Hindi + Indian-English (Hinglish) Code-Switching.**  
> Built for ThinkNCollab and open-sourced for the community. Auto-binds to any server IP on any available port, providing a 100% OpenAI Whisper API specification drop-in replacement.

---

## Key Features

- **Open-Source and 100% Free**: No API keys, no monthly fees, 100% offline local inference.
- **Hinglish Code-Switching Support**: High accuracy for Hindi (Devanagari) + Indian English mixed meeting speech.
- **Automatic Server Binding (0.0.0.0) & Auto-Port Detection**: Runs on any Linux, Mac, or Windows server and auto-detects an open port.
- **100% OpenAI Whisper API Specification Compatible**: Works seamlessly with official OpenAI JS/Python SDKs by setting `baseURL: "http://<SERVER_IP>:8000/v1"`.
- **Spectral Subtraction Noise Gate**: Automatic background noise reduction for fan, AC hum, and street noise.
- **Official ThinkNCollab JavaScript SDK (@thinkncollab/scribe) & Python CLI (thinkncollab_whisper.py)**: Instant integration into Node.js, Express, React, Next.js, and Python.

---

## 1. Quickstart Server Execution

Run the server on ANY host machine or cloud server (AWS, GCP, DigitalOcean, Mac):

```bash
# Clone Repository & Install Dependencies
git clone https://github.com/ThinkNCollab/ThinkNCollab_TRANSCRIPT.git
cd ThinkNCollab_TRANSCRIPT

pip install -r requirements.txt

# Start Server (Auto-binds open port on 0.0.0.0)
python3 server.py
```

### Server Output Example:

```text
==========================================================================
  ThinkNCollab ASR Server Active on Server Host: 0.0.0.0
==========================================================================
  • Server IP / Host   : http://192.168.1.15:8000
  • OpenAI API Endpoint: http://192.168.1.15:8000/v1/audio/transcriptions
  • Network Binding    : 0.0.0.0 (Accessible across server network/internet)
==========================================================================
```

---

## 2. OpenAI JS SDK Integration (ThinkNCollab / Node.js)

Drop-in replacement for OpenAI's official `openai` NPM package (`npm install openai`):

```javascript
import OpenAI from 'openai';

// 1. Point OpenAI SDK to your ThinkNCollab Server IP/Port:
const openai = new OpenAI({
    baseURL: 'http://<YOUR_SERVER_IP>:8000/v1',
    apiKey: 'thinkncollab-free-key'
});

// 2. Transcribe Audio (Identical to Official OpenAI Whisper syntax)
async function transcribeMeeting(audioStream) {
    const transcript = await openai.audio.transcriptions.create({
        model: 'whisper-1',
        file: audioStream,
        language: 'hinglish'
    });

    console.log("ThinkNCollab Transcript:", transcript.text);
    return transcript.text;
}
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

## 4. Terminal CLI & Python Library Usage

Use ThinkNCollab-Whisper from command line or inside Python applications:

```bash
# Terminal CLI Command
python3 thinkncollab_whisper.py meeting.wav --language hinglish --output_format txt
```

```python
# Python Library Import
import thinkncollab_whisper

model = thinkncollab_whisper.load_model("small")
result = model.transcribe("meeting.wav", language="hinglish")
print(result["text"])
```

---

## Project Directory Layout

```text
ThinkNCollab_TRANSCRIPT/
├── README.md                  # Production Open-Source Documentation
├── server.py                  # Auto-Port OpenAI API Compatible Server (0.0.0.0 Host)
├── thinkncollab_whisper.py    # OpenAI Whisper Style CLI & Python Library Script
├── thinkncollab-scribe-sdk.js # Official ThinkNCollab JavaScript / TypeScript SDK Client
├── package.json               # NPM Package Manifest (@thinkncollab/scribe)
├── setup.py                   # PyPI Package Manifest (thinkncollab-scribe)
├── requirements.txt           # Core Python Dependencies
├── checkpoints/               # Trained Model Weights (whisper_small_hinglish_final)
├── src/                       # Core Engine Modules (Noise Reducer, Model Loaders)
└── notebooks/                 # Kaggle CUDA Training Notebook
```

---

## License

MIT License &copy; 2026 ThinkNCollab AI Team. Open-Source for the community.
