# VibeVoice TTS API

REST API for multi-speaker text-to-speech with streaming and download support.

## Quick Start

```bash
cd /Users/hareesh/dev/ai/VibeVoice

# Activate the virtual environment
source .venv/bin/activate

# Start server in foreground (port 8001, MPS device)
./start-tts-server.sh

# Or with custom options
./start-tts-server.sh --port 9000 --device cpu
```

## Setup (First Time)

```bash
cd /Users/hareesh/dev/ai/VibeVoice

# Create venv and install dependencies using uv
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Download model (if not already cached)
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B
```

## Server Script Options

```
./start-tts-server.sh [OPTIONS]

  --port PORT         Port to listen on (default: 8001)
  --device DEVICE     Compute device: mps | cuda | cpu (default: mps)
  --model-path PATH   Path to the VibeVoice model
  --daemon            Run in background as a daemon
  --stop              Stop a running daemon
  --status            Check if daemon is running
```

### Run as background daemon

```bash
# Start
./start-tts-server.sh --daemon

# Check status
./start-tts-server.sh --status

# Stop
./start-tts-server.sh --stop
```

### Run as a persistent macOS system service (survives reboots)

```bash
# Install the LaunchAgent
cp com.vibevoice.tts-server.plist ~/Library/LaunchAgents/

# Load and start
launchctl load ~/Library/LaunchAgents/com.vibevoice.tts-server.plist

# Check status
launchctl list | grep vibevoice

# Stop and unload
launchctl unload ~/Library/LaunchAgents/com.vibevoice.tts-server.plist
```

---

## API Endpoints

| Method | Path                    | Body Type  | Description                                  |
| ------ | ----------------------- | ---------- | -------------------------------------------- |
| GET    | `/api/health`           | —          | Health check                                 |
| GET    | `/api/voices`           | —          | List available voice presets                 |
| POST   | `/api/tts`              | JSON       | Generate & download audio (structured JSON)  |
| POST   | `/api/tts/text`         | JSON       | Generate & download audio (tagged text JSON) |
| POST   | `/api/tts/plain`        | Plain text | Generate & download audio (plain text)       |
| POST   | `/api/tts/stream`       | JSON       | Stream audio chunks (structured JSON)        |
| POST   | `/api/tts/stream/text`  | JSON       | Stream audio chunks (tagged text JSON)       |
| POST   | `/api/tts/stream/plain` | Plain text | Stream audio chunks (plain text)             |

### Interactive docs

Once the server is running, open:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## API Usage Examples

### 1. List available voices

```bash
curl http://localhost:8001/api/voices | python3 -m json.tool
```

Response:

```json
{
  "voices": [
    "de-Spk0_man",
    "de-Spk1_woman",
    "en-Carter_man",
    "en-Davis_man",
    "en-Emma_woman",
    "en-Frank_man",
    "en-Grace_woman",
    "en-Mike_man",
    "fr-Spk0_man",
    "..."
  ],
  "default_voice": "en-Carter_man",
  "total": 25
}
```

### 2. Generate & download audio (multi-speaker, structured JSON)

```bash
curl -X POST http://localhost:8001/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"speaker": "en-Carter_man", "text": "Hello, how are you doing today?"},
      {"speaker": "en-Emma_woman", "text": "I am doing great, thanks for asking!"},
      {"speaker": "en-Carter_man", "text": "Wonderful. Let me tell you about the weather."}
    ],
    "cfg_scale": 1.5,
    "format": "wav"
  }' \
  --output output.wav
```

### 3. Generate & download audio (tagged text)

```bash
curl -X POST http://localhost:8001/api/tts/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[en-Carter_man] Hello! Welcome to our podcast. [en-Emma_woman] Thanks for having me! It is great to be here.",
    "format": "wav"
  }' \
  --output output.wav
```

### 4. Stream audio (for real-time playback)

```bash
# Stream to file (starts writing immediately as audio is generated)
curl -X POST http://localhost:8001/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"speaker": "en-Carter_man", "text": "This audio is being streamed in real-time."},
      {"speaker": "en-Emma_woman", "text": "And now it is my turn to speak."}
    ]
  }' \
  --output stream_output.wav

# Stream tagged text
curl -X POST http://localhost:8001/api/tts/stream/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[en-Mike_man] Streaming is great for long content because you hear audio before the full generation completes."
  }' \
  --output stream_output.wav
```

### 5. Plain text input (no JSON needed)

Send raw text with `[SpeakerName]` tags directly as the request body:

```bash
# Download
curl -X POST http://localhost:8001/api/tts/plain \
  -H "Content-Type: text/plain" \
  -d '[en-Carter_man] Hello! Welcome to our podcast. [en-Emma_woman] Thanks for having me!' \
  --output output.wav

# Stream
curl -X POST http://localhost:8001/api/tts/stream/plain \
  -H "Content-Type: text/plain" \
  -d '[en-Carter_man] Streaming with plain text is the simplest way to use the API.' \
  --output stream.wav

# With optional query params
curl -X POST "http://localhost:8001/api/tts/plain?cfg_scale=2.0&steps=8&format=wav" \
  -H "Content-Type: text/plain" \
  -d '[en-Mike_man] Custom cfg scale and steps.' \
  --output output.wav

# Single speaker (no tags) uses the default voice
curl -X POST http://localhost:8001/api/tts/plain \
  -H "Content-Type: text/plain" \
  -d 'Just plain text with no speaker tags at all.' \
  --output default_voice.wav
```

### 6. Pipe streamed audio directly to a player (macOS)

```bash
curl -sN -X POST http://localhost:8001/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"speaker": "en-Carter_man", "text": "You can hear me as I am being generated!"}
    ]
  }' | afplay -
```

### 7. Single speaker (no speaker tag needed)

```bash
curl -X POST http://localhost:8001/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {"text": "Hello world, this uses the default voice."}
    ]
  }' \
  --output single_speaker.wav
```

### 8. Python client example

```python
import requests

# Download complete audio
response = requests.post("http://localhost:8001/api/tts", json={
    "segments": [
        {"speaker": "en-Carter_man", "text": "Hello from Python!"},
        {"speaker": "en-Emma_woman", "text": "Nice to meet you!"},
    ],
    "cfg_scale": 1.5,
    "format": "wav",
})
with open("output.wav", "wb") as f:
    f.write(response.content)

# Stream audio
response = requests.post("http://localhost:8001/api/tts/stream", json={
    "segments": [
        {"speaker": "en-Carter_man", "text": "Streaming from Python!"},
    ],
}, stream=True)
with open("streamed.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=4096):
        f.write(chunk)
```

---

## Request Body Reference

### `POST /api/tts` and `POST /api/tts/stream`

```json
{
  "segments": [
    { "speaker": "en-Carter_man", "text": "Text for this speaker" },
    { "speaker": "en-Emma_woman", "text": "Text for another speaker" }
  ],
  "cfg_scale": 1.5,
  "inference_steps": 5,
  "do_sample": false,
  "temperature": 0.9,
  "top_p": 0.9,
  "format": "wav"
}
```

| Field                | Type   | Default    | Description                                            |
| -------------------- | ------ | ---------- | ------------------------------------------------------ |
| `segments`           | array  | _required_ | List of `{speaker?, text}` objects                     |
| `segments[].speaker` | string | `null`     | Voice preset name. Omit for default voice.             |
| `segments[].text`    | string | _required_ | Text for this speaker to say.                          |
| `cfg_scale`          | float  | `1.5`      | Classifier-free guidance scale (0.1–10.0)              |
| `inference_steps`    | int    | `null`     | Diffusion inference steps (1–50, null = model default) |
| `do_sample`          | bool   | `false`    | Enable sampling in generation                          |
| `temperature`        | float  | `0.9`      | Sampling temperature (0.0–2.0)                         |
| `top_p`              | float  | `0.9`      | Top-p sampling (0.0–1.0)                               |
| `format`             | string | `"wav"`    | Output format: `"wav"` or `"pcm"`                      |

### `POST /api/tts/text` and `POST /api/tts/stream/text`

```json
{
  "text": "[en-Carter_man] Hello! [en-Emma_woman] Hi there!",
  "cfg_scale": 1.5,
  "format": "wav"
}
```

| Field                        | Type   | Default    | Description                              |
| ---------------------------- | ------ | ---------- | ---------------------------------------- |
| `text`                       | string | _required_ | Text with optional `[SpeakerName]` tags. |
| (other fields same as above) |

---

## Stream vs Download

| Feature       | `/api/tts` (download)         | `/api/tts/stream` (stream)      |
| ------------- | ----------------------------- | ------------------------------- |
| Latency       | Must wait for full generation | Audio starts immediately        |
| WAV header    | Correct file length           | Placeholder length (0xFFFFFFFF) |
| Best for      | Short text, file saving       | Long text, real-time playback   |
| `curl` output | Save to file with `--output`  | Pipe to player or save          |

---

## Troubleshooting

- **Port already in use**: `./start-tts-server.sh --stop` or `lsof -i :8001`
- **Model not found**: Set `--model-path` or `MODEL_PATH` env var
- **MPS errors on Mac**: Try `--device cpu` as fallback
- **Check logs**: `tail -f tts-server.log` (daemon mode) or `tts-server-launchd.log` (launchd)
