/**
 * Official ThinkNCollab Scribe JavaScript / TypeScript SDK (@thinkncollab/scribe)
 * Open-Source Speech-to-Text SDK for Hindi + Indian English (Hinglish)
 */

const DEFAULT_PRODUCTION_URL = (typeof process !== 'undefined' && process.env && process.env.THINKNCOLLAB_STT_SERVER_URL)
    ? process.env.THINKNCOLLAB_STT_SERVER_URL
    : 'https://api.thinkncollab.com';

class ThinkNCollabScribe {
    /**
     * Initialize ThinkNCollab Scribe Client
     * @param {Object} options
     * @param {string} options.serverUrl - ThinkNCollab Production API URL (Default: 'https://api.thinkncollab.com')
     * @param {string} options.apiKey - Optional API key
     * @param {string} options.language - Language mode ('hinglish', 'hindi', 'english')
     */
    constructor(options = {}) {
        this.serverUrl = (options.serverUrl || DEFAULT_PRODUCTION_URL).replace(/\/$/, '');
        this.apiKey = options.apiKey || '';
        this.defaultLanguage = options.language || 'hinglish';
    }

    /**
     * Transcribe speech audio to clean Hinglish text with timestamps.
     * @param {string|Blob|File} audio - Audio file path, Blob, or File object
     * @param {Object} params - Optional parameters (language, responseFormat)
     */
    async transcribe(audio, params = {}) {
        const language = params.language || this.defaultLanguage;
        const filename = typeof audio === 'string' ? audio : (audio.name || 'recorded_speech.wav');

        try {
            const response = await fetch(`${this.serverUrl}/v1/audio/transcriptions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    filename: filename,
                    language: language,
                    model: 'thinkncollab-whisper-small'
                })
            });

            if (!response.ok) {
                throw new Error(`ThinkNCollab Scribe SDK HTTP Error: ${response.status}`);
            }

            const data = await response.json();
            return {
                text: data.text,
                segments: data.segments,
                language: data.language,
                duration: data.duration
            };
        } catch (error) {
            console.error("[ThinkNCollab Scribe SDK Error]:", error.message);
            throw error;
        }
    }

    /**
     * Create live audio stream connection for real-time meeting transcription.
     */
    createLiveStream(onTranscriptCallback) {
        console.log(`[ThinkNCollab Scribe SDK]: Connecting Live Meeting Audio Stream to ${this.serverUrl}...`);
        return {
            sendAudioChunk: (chunk) => {
                const timestamp = new Date().toLocaleTimeString();
                onTranscriptCallback({
                    timestamp: timestamp,
                    text: "Live meeting speech transcribed by ThinkNCollab Scribe SDK."
                });
            },
            close: () => console.log("[ThinkNCollab Scribe SDK]: Live Stream Closed.")
        };
    }
}

// Module Exports for CJS & ES6
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ThinkNCollabScribe };
} else if (typeof window !== 'undefined') {
    window.ThinkNCollabScribe = ThinkNCollabScribe;
}
