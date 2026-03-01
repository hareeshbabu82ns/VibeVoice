#!/usr/bin/env bash
# ============================================================================
#  VibeVoice TTS API Server — Startup Script
#
#  Usage:
#    ./start-tts-server.sh                  # foreground (default port 8001, mps)
#    ./start-tts-server.sh --port 9000      # custom port
#    ./start-tts-server.sh --device cuda    # use CUDA GPU
#    ./start-tts-server.sh --daemon         # run in background as a daemon
#    ./start-tts-server.sh --stop           # stop a running daemon
#    ./start-tts-server.sh --status         # check if daemon is running
# ============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PORT=8001
DEVICE="mps"
MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4}"
VOICES_DIR="${VOICES_DIR:-}"
DAEMON=false
STOP=false
STATUS=false
PID_FILE="${SCRIPT_DIR}/.tts-server.pid"
LOG_FILE="${SCRIPT_DIR}/tts-server.log"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)       PORT="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --voices-dir) VOICES_DIR="$(cd "$2" && pwd)"; shift 2 ;;
        --daemon)     DAEMON=true; shift ;;
        --stop)       STOP=true; shift ;;
        --status)     STATUS=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--port PORT] [--device DEVICE] [--model-path PATH] [--voices-dir PATH] [--daemon] [--stop] [--status]"
            echo ""
            echo "Options:"
            echo "  --port PORT         Port to listen on (default: 8001)"
            echo "  --device DEVICE     Compute device: mps | cuda | cpu (default: mps)"
            echo "  --model-path PATH   Path to VibeVoice model (default: HuggingFace cache)"
            echo "  --voices-dir PATH   Directory containing voice preset .pt files"
            echo "                      (default: demo/voices/streaming_model)"
            echo "                      Use when running a model that needs different presets,"
            echo "                      e.g. VibeVoice-1.5B requires its own presets."
            echo "  --daemon            Run in background and keep running"
            echo "  --stop              Stop a running daemon"
            echo "  --status            Check if daemon is running"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Stop command ──────────────────────────────────────────────────────────────
if $STOP; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping TTS server (PID $PID)..."
            kill "$PID"
            # Wait up to 10s for graceful shutdown
            for i in $(seq 1 10); do
                if ! kill -0 "$PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                echo "Force killing PID $PID..."
                kill -9 "$PID"
            fi
            rm -f "$PID_FILE"
            echo "Server stopped."
        else
            echo "PID $PID not running. Cleaning up stale PID file."
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found. Server may not be running."
    fi
    exit 0
fi

# ── Status command ────────────────────────────────────────────────────────────
if $STATUS; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "TTS server is running (PID $PID)"
            echo "Log file: $LOG_FILE"
            exit 0
        else
            echo "PID file exists but process $PID is not running."
            rm -f "$PID_FILE"
            exit 1
        fi
    else
        echo "TTS server is not running."
        exit 1
    fi
fi

# ── Validate environment ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Create it with:  uv venv .venv && uv pip install -e ."
    exit 1
fi

PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

# ── Export environment ────────────────────────────────────────────────────────
export MODEL_PATH
export MODEL_DEVICE="$DEVICE"
[[ -n "$VOICES_DIR" ]] && export VOICES_DIR

echo "=============================================="
echo "  VibeVoice TTS API Server"
echo "=============================================="
echo "  Model:   $MODEL_PATH"
echo "  Device:  $DEVICE"
echo "  Port:    $PORT"
[[ -n "$VOICES_DIR" ]] && echo "  Voices:  $VOICES_DIR"
echo "  Python:  $PYTHON"
echo "=============================================="

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

CMD=("$PYTHON" "-m" "uvicorn" "web.app:app" "--host" "0.0.0.0" "--port" "$PORT")

if $DAEMON; then
    # Check if already running
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Server is already running (PID $OLD_PID). Use --stop first."
            exit 1
        fi
        rm -f "$PID_FILE"
    fi

    echo "Starting in daemon mode..."
    echo "  Log file: $LOG_FILE"
    echo "  PID file: $PID_FILE"

    # Launch in background with nohup
    nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$PID_FILE"

    # Wait a moment to verify it started
    sleep 3
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "Server started successfully (PID $DAEMON_PID)"
        echo ""
        echo "API endpoints:"
        echo "  Health:            http://localhost:${PORT}/api/health"
        echo "  Voices:            http://localhost:${PORT}/api/voices"
        echo "  TTS (download):    POST http://localhost:${PORT}/api/tts"
        echo "  TTS (stream):      POST http://localhost:${PORT}/api/tts/stream"
        echo "  TTS text (download): POST http://localhost:${PORT}/api/tts/text"
        echo "  TTS text (stream):   POST http://localhost:${PORT}/api/tts/stream/text"
        echo ""
        echo "To stop:   $0 --stop"
        echo "To check:  $0 --status"
    else
        echo "ERROR: Server failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
else
    # Foreground mode — run directly; Ctrl-C to stop
    echo "Starting in foreground (Ctrl-C to stop)..."
    echo ""
    cd demo
    exec "${CMD[@]}"
fi
