#!/usr/bin/env python3
import os
import sys
import json
import time
import socket
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import thinkncollab_whisper
except ImportError:
    from src.scribe_engine import AIScribeTranscriberEngine
    thinkncollab_whisper = None

DEFAULT_PORT = int(os.environ.get("PORT", 8000))
model_instance = thinkncollab_whisper.load_model("small") if thinkncollab_whisper else None

def get_server_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def find_available_port(start_port=8000):
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                return port
            port += 1
    return start_port

class ThinkNCollabAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        clean_path = self.path.split("?")[0].rstrip("/")

        if clean_path in ["/api/status", "/v1/models"]:
            response = {
                "object": "list",
                "data": [
                    {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "thinkncollab-ai"},
                    {"id": "thinkncollab-whisper-small", "object": "model", "created": 1677610602, "owned_by": "thinkncollab-ai"}
                ]
            }
            res_bytes = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(res_bytes)
            return

        if clean_path in ["/api/transcribe", "/v1/audio/transcriptions", "/v1/audio/translations"]:
            response = {
                "status": "ONLINE",
                "service": "ThinkNCollab ASR Speech Transcriber API",
                "endpoint": clean_path,
                "method": "POST"
            }
            res_bytes = json.dumps(response, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(res_bytes)
            return

        response = {
            "status": "ONLINE",
            "server": "ThinkNCollab ASR API",
            "version": "1.0.1"
        }
        res_bytes = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(res_bytes)))
        self.end_headers()
        self.wfile.write(res_bytes)

    def do_POST(self):
        clean_path = self.path.split("?")[0].rstrip("/")

        if clean_path in ["/v1/audio/transcriptions", "/v1/audio/translations", "/api/transcribe"]:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b""

            filename = "recorded_mic_audio.wav"
            spoken_text = None
            task = "translate" if "translations" in clean_path else "transcribe"

            try:
                if post_data.startswith(b"{"):
                    payload = json.loads(post_data.decode("utf-8"))
                    raw_fn = payload.get("filename", filename)
                    spoken_text = payload.get("text", None)
                    if isinstance(raw_fn, dict):
                        filename = str(raw_fn.get("name", filename))
                    elif isinstance(raw_fn, str):
                        filename = raw_fn
                    else:
                        filename = str(raw_fn)
            except Exception:
                filename = "recorded_mic_audio.wav"

            if model_instance:
                res_dict = model_instance.transcribe(filename, task=task, verbose=True)
                full_text = res_dict["text"]
                segments = res_dict["segments"]
            else:
                full_text = f"[{time.strftime('%M:%S')}] Speaker 1: Audio processed by ThinkNCollab server."
                segments = [{"id": 0, "start": 0.0, "end": 5.0, "text": full_text, "speaker": "Speaker 1"}]

            openai_response = {
                "text": full_text,
                "segments": segments,
                "language": "hinglish",
                "duration": 30.0
            }

            res_bytes = json.dumps(openai_response, ensure_ascii=False).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(res_bytes)
            return

        if self.command == "OPTIONS":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Connection", "close")
            self.end_headers()
            return

        self.send_error(404, "Endpoint not found")

def run_server(port=None):
    if port is None:
        port = find_available_port(DEFAULT_PORT)

    host = "0.0.0.0"
    server_ip = get_server_ip()

    server_address = (host, port)
    httpd = HTTPServer(server_address, ThinkNCollabAPIHandler)

    print("==========================================================================")
    print(f"  ThinkNCollab ASR API Server Active on Host: {host}")
    print("==========================================================================")
    print(f"  - Server IP / Host   : http://{server_ip}:{port}")
    print(f"  - Endpoint           : http://{server_ip}:{port}/v1/audio/transcriptions")
    print(f"  - Network Binding    : 0.0.0.0")
    print("==========================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThinkNCollab ASR API Server")
    parser.add_argument("--port", type=int, default=None, help="Port to bind server")
    args = parser.parse_args()

    run_server(port=args.port)
