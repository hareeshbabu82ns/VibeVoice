import datetime
import builtins
import asyncio
import io
import json
import os
import struct
import threading
import traceback
import wave
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, Request, Body, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

import copy
import re

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def parse_multi_speaker_text(text: str) -> list:
    """Parse text with [SpeakerName] tags into a list of (speaker, text) tuples.

    Format:
        [en-Carter_man] Hello, how are you?
        [en-Emma_woman] I'm doing great, thanks!

    If no speaker tags are found, returns a single segment with speaker=None.
    """
    pattern = re.compile(r"\[([^\]]+)\]")
    segments = []
    last_end = 0
    current_speaker = None

    for match in pattern.finditer(text):
        # Capture any text before this tag that belongs to the previous speaker
        preceding_text = text[last_end:match.start()].strip()
        if preceding_text and current_speaker is not None:
            segments.append((current_speaker, preceding_text))
        elif preceding_text and current_speaker is None:
            # Text before the very first tag — treat as untagged
            segments.append((None, preceding_text))

        current_speaker = match.group(1).strip()
        last_end = match.end()

    # Remaining text after the last tag
    remaining = text[last_end:].strip()
    if remaining:
        segments.append((current_speaker, remaining))

    # If nothing was parsed (no tags at all), return the whole text as one segment
    if not segments:
        stripped = text.strip()
        if stripped:
            segments.append((None, stripped))

    return segments


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-Carter_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def stream_multi_speaker(
        self,
        segments: list,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        default_voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[Tuple[Optional[str], np.ndarray]]:
        """Generate audio for multiple speaker segments sequentially.

        Yields (speaker_name, audio_chunk) tuples so the caller can
        track which speaker is currently being synthesized.
        """
        stop_signal = stop_event or threading.Event()

        for idx, (speaker, text) in enumerate(segments):
            if stop_signal.is_set():
                break

            voice = speaker if speaker and speaker in self.voice_presets else default_voice_key

            if log_callback:
                log_callback(
                    "speaker_change",
                    speaker=speaker or "default",
                    voice=voice,
                    segment_index=idx,
                    total_segments=len(segments),
                )

            # Each segment gets its own per-segment stop event so that
            # the stream() finally-block (which calls stop_signal.set())
            # does not abort the remaining segments.  We propagate the
            # outer stop_signal into it manually.
            segment_stop = threading.Event()

            for chunk in self.stream(
                text=text,
                cfg_scale=cfg_scale,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                refresh_negative=refresh_negative,
                inference_steps=inference_steps,
                voice_key=voice,
                log_callback=log_callback,
                stop_event=segment_stop,
            ):
                if stop_signal.is_set():
                    segment_stop.set()
                    break
                yield (speaker, chunk)

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")
    
    service = StreamingTTSService(
        model_path=model_path,
        device=device
    )
    service.load()

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()
    print("[startup] Model ready.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)


def streaming_tts_multi(segments: list, **kwargs) -> Iterator[Tuple[Optional[str], np.ndarray]]:
    """Yield (speaker, audio_chunk) for multi-speaker segments."""
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream_multi_speaker(segments, **kwargs)


def _is_multi_speaker(text: str) -> bool:
    """Return True if the text contains [SpeakerName] tags."""
    return bool(re.search(r"\[[^\]]+\]", text))


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
    except ValueError:
        cfg_scale = 1.5
    if cfg_scale <= 0:
        cfg_scale = 1.5
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    # Detect multi-speaker mode
    multi_speaker = _is_multi_speaker(text)
    segments = parse_multi_speaker_text(text) if multi_speaker else []
    if multi_speaker:
        print(f"[multi-speaker] Detected {len(segments)} segment(s): "
              + ", ".join(f"{s or 'default'}" for s, _ in segments))

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
            multi_speaker=multi_speaker,
            segments=len(segments) if multi_speaker else 1,
        )

        stop_signal = threading.Event()

        if multi_speaker:
            iterator = streaming_tts_multi(
                segments,
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
                default_voice_key=voice_param or service.default_voice_key,
                log_callback=enqueue_log,
                stop_event=stop_signal,
            )
        else:
            # Wrap single-speaker iterator to match (speaker, chunk) shape
            _inner = streaming_tts(
                text,
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
                voice_key=voice_param,
                log_callback=enqueue_log,
                stop_event=stop_signal,
            )
            iterator = ((None, chunk) for chunk in _inner)

        sentinel = object()
        first_ws_send_logged = False

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                await flush_logs()
                result = await asyncio.to_thread(next, iterator, sentinel)
                if result is sentinel:
                    break
                speaker, chunk = result
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        except Exception as e:
            print(f"Error in websocket stream: {e}")
            traceback.print_exc()
            enqueue_log("backend_error", message=str(e))
            stop_signal.set()
        finally:
            stop_signal.set()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
    }


# ---------------------------------------------------------------------------
#  REST API – Pydantic models
# ---------------------------------------------------------------------------

class SpeakerSegment(BaseModel):
    """A single speaker segment."""
    speaker: Optional[str] = Field(
        None,
        description="Voice preset name (e.g. 'en-Carter_man'). Omit to use the default voice.",
    )
    text: str = Field(..., description="Text for this speaker to say.")


class TTSRequest(BaseModel):
    """JSON body for the TTS API."""
    segments: List[SpeakerSegment] = Field(
        ...,
        description="Ordered list of speaker segments.",
        min_length=1,
    )
    cfg_scale: float = Field(1.5, ge=0.1, le=10.0, description="Classifier-free guidance scale.")
    inference_steps: Optional[int] = Field(None, ge=1, le=50, description="Diffusion inference steps.")
    do_sample: bool = Field(False, description="Enable sampling in generation.")
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    format: str = Field("wav", description="Output format: 'wav' or 'pcm'.")


class TTSTextRequest(BaseModel):
    """Alternative: send a single text with [Speaker] tags."""
    text: str = Field(..., description="Text with optional [SpeakerName] tags.")
    cfg_scale: float = Field(1.5, ge=0.1, le=10.0)
    inference_steps: Optional[int] = Field(None, ge=1, le=50)
    do_sample: bool = Field(False)
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    format: str = Field("wav", description="Output format: 'wav' or 'pcm'.")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _collect_all_audio(
    service: StreamingTTSService,
    segments: List[Tuple[Optional[str], str]],
    cfg_scale: float,
    inference_steps: Optional[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> np.ndarray:
    """Run multi-speaker generation and return the complete audio array."""
    chunks: List[np.ndarray] = []
    for _speaker, chunk in service.stream_multi_speaker(
        segments,
        cfg_scale=cfg_scale,
        inference_steps=inference_steps,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    ):
        chunks.append(chunk)
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array to a WAV file in memory."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_header(sample_rate: int = SAMPLE_RATE, bits: int = 16, channels: int = 1) -> bytes:
    """Create a WAV header for streaming (unknown length → 0xFFFFFFFF)."""
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    # Use max uint32 for data size to signal streaming
    data_size = 0xFFFFFFFF
    file_size = 36 + data_size  # will overflow for uint32, that's OK for streaming
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size & 0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,            # chunk size
        1,             # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size & 0xFFFFFFFF,
    )
    return header


def _segments_from_request(req: TTSRequest) -> List[Tuple[Optional[str], str]]:
    return [(seg.speaker, seg.text) for seg in req.segments]


def _segments_from_text_request(req: TTSTextRequest) -> List[Tuple[Optional[str], str]]:
    return parse_multi_speaker_text(req.text)


# ---------------------------------------------------------------------------
#  POST /api/tts  –  Full download (WAV or raw PCM)
# ---------------------------------------------------------------------------

@app.post("/api/tts", summary="Generate TTS audio (full download)")
async def api_tts(req: TTSRequest):
    """Accept a JSON list of speaker segments and return the complete audio file.

    Example request:
    ```json
    {
      "segments": [
        {"speaker": "en-Carter_man", "text": "Hello, how are you?"},
        {"speaker": "en-Emma_woman", "text": "I am doing great, thanks!"}
      ],
      "cfg_scale": 1.5,
      "format": "wav"
    }
    ```
    """
    service: StreamingTTSService = app.state.tts_service
    segments = _segments_from_request(req)

    if not segments:
        raise HTTPException(status_code=400, detail="No text segments provided.")

    audio = await asyncio.to_thread(
        _collect_all_audio,
        service,
        segments,
        req.cfg_scale,
        req.inference_steps,
        req.do_sample,
        req.temperature,
        req.top_p,
    )

    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio generated.")

    if req.format == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.pcm",
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Channels": "1",
                "X-Bits-Per-Sample": "16",
            },
        )

    wav_bytes = _audio_to_wav_bytes(audio)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


# ---------------------------------------------------------------------------
#  POST /api/tts/text  –  Tagged text input (full download)
# ---------------------------------------------------------------------------

@app.post("/api/tts/text", summary="Generate TTS from tagged text (full download)")
async def api_tts_text(req: TTSTextRequest):
    """Accept text with [SpeakerName] tags and return the complete audio file.

    Example request:
    ```json
    {
      "text": "[en-Carter_man] Hello! [en-Emma_woman] Hi there!",
      "format": "wav"
    }
    ```
    """
    service: StreamingTTSService = app.state.tts_service
    segments = _segments_from_text_request(req)

    if not segments:
        raise HTTPException(status_code=400, detail="No text provided.")

    audio = await asyncio.to_thread(
        _collect_all_audio,
        service,
        segments,
        req.cfg_scale,
        req.inference_steps,
        req.do_sample,
        req.temperature,
        req.top_p,
    )

    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio generated.")

    if req.format == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.pcm",
                "X-Sample-Rate": str(SAMPLE_RATE),
            },
        )

    wav_bytes = _audio_to_wav_bytes(audio)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


# ---------------------------------------------------------------------------
#  POST /api/tts/stream  –  Streaming chunked WAV audio
# ---------------------------------------------------------------------------

@app.post("/api/tts/stream", summary="Stream TTS audio (chunked)")
async def api_tts_stream(req: TTSRequest):
    """Accept a JSON list of speaker segments and stream audio chunks back.

    Returns a chunked WAV stream. The first chunk is the WAV header,
    followed by PCM audio chunks as they are generated.
    """
    service: StreamingTTSService = app.state.tts_service
    segments = _segments_from_request(req)

    if not segments:
        raise HTTPException(status_code=400, detail="No text segments provided.")

    def _generate():
        # Yield WAV header first (streaming-style with unknown length)
        yield _wav_header()

        for _speaker, chunk in service.stream_multi_speaker(
            segments,
            cfg_scale=req.cfg_scale,
            inference_steps=req.inference_steps,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_p=req.top_p,
        ):
            chunk = np.clip(chunk, -1.0, 1.0)
            pcm = (chunk * 32767.0).astype(np.int16)
            yield pcm.tobytes()

    return StreamingResponse(
        _generate(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=tts_stream.wav",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "Transfer-Encoding": "chunked",
        },
    )


# ---------------------------------------------------------------------------
#  POST /api/tts/stream/text  –  Streaming from tagged text
# ---------------------------------------------------------------------------

@app.post("/api/tts/stream/text", summary="Stream TTS from tagged text (chunked)")
async def api_tts_stream_text(req: TTSTextRequest):
    """Accept text with [SpeakerName] tags and stream audio chunks back."""
    service: StreamingTTSService = app.state.tts_service
    segments = _segments_from_text_request(req)

    if not segments:
        raise HTTPException(status_code=400, detail="No text provided.")

    def _generate():
        yield _wav_header()

        for _speaker, chunk in service.stream_multi_speaker(
            segments,
            cfg_scale=req.cfg_scale,
            inference_steps=req.inference_steps,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_p=req.top_p,
        ):
            chunk = np.clip(chunk, -1.0, 1.0)
            pcm = (chunk * 32767.0).astype(np.int16)
            yield pcm.tobytes()

    return StreamingResponse(
        _generate(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=tts_stream.wav",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "Transfer-Encoding": "chunked",
        },
    )


# ---------------------------------------------------------------------------
#  GET /api/voices  –  List available voices
# ---------------------------------------------------------------------------

@app.get("/api/voices", summary="List available voice presets")
def api_voices():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
        "total": len(voices),
    }


# ---------------------------------------------------------------------------
#  POST /api/tts/plain  –  Plain text input (full download)
# ---------------------------------------------------------------------------

@app.post("/api/tts/plain", summary="Generate TTS from plain text with [Speaker] tags (download)")
async def api_tts_plain(request: Request):
    """Accept raw plain text with [SpeakerName] tags and return audio.

    Send the body as text/plain. Optional query params: cfg_scale, steps, format.

    Example:
        curl -X POST http://localhost:8001/api/tts/plain \\
          -H "Content-Type: text/plain" \\
          -d '[en-Carter_man] Hello! [en-Emma_woman] Hi there!' \\
          --output output.wav
    """
    service: StreamingTTSService = app.state.tts_service
    body = (await request.body()).decode("utf-8").strip()

    if not body:
        raise HTTPException(status_code=400, detail="Empty body.")

    segments = parse_multi_speaker_text(body)
    if not segments:
        raise HTTPException(status_code=400, detail="No text segments found.")

    cfg_scale = float(request.query_params.get("cfg_scale", "1.5"))
    inference_steps_raw = request.query_params.get("steps")
    inference_steps = int(inference_steps_raw) if inference_steps_raw else None
    fmt = request.query_params.get("format", "wav")

    audio = await asyncio.to_thread(
        _collect_all_audio,
        service,
        segments,
        cfg_scale,
        inference_steps,
        False,   # do_sample
        0.9,     # temperature
        0.9,     # top_p
    )

    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio generated.")

    if fmt == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.pcm",
                "X-Sample-Rate": str(SAMPLE_RATE),
            },
        )

    wav_bytes = _audio_to_wav_bytes(audio)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


# ---------------------------------------------------------------------------
#  POST /api/tts/stream/plain  –  Plain text input (streaming)
# ---------------------------------------------------------------------------

@app.post("/api/tts/stream/plain", summary="Stream TTS from plain text with [Speaker] tags")
async def api_tts_stream_plain(request: Request):
    """Accept raw plain text with [SpeakerName] tags and stream audio back.

    Example:
        curl -X POST http://localhost:8001/api/tts/stream/plain \\
          -H "Content-Type: text/plain" \\
          -d '[en-Carter_man] Hello! [en-Emma_woman] Hi there!' \\
          --output stream.wav
    """
    service: StreamingTTSService = app.state.tts_service
    body = (await request.body()).decode("utf-8").strip()

    if not body:
        raise HTTPException(status_code=400, detail="Empty body.")

    segments = parse_multi_speaker_text(body)
    if not segments:
        raise HTTPException(status_code=400, detail="No text segments found.")

    cfg_scale = float(request.query_params.get("cfg_scale", "1.5"))
    inference_steps_raw = request.query_params.get("steps")
    inference_steps = int(inference_steps_raw) if inference_steps_raw else None

    def _generate():
        yield _wav_header()

        for _speaker, chunk in service.stream_multi_speaker(
            segments,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
        ):
            chunk = np.clip(chunk, -1.0, 1.0)
            pcm = (chunk * 32767.0).astype(np.int16)
            yield pcm.tobytes()

    return StreamingResponse(
        _generate(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=tts_stream.wav",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "Transfer-Encoding": "chunked",
        },
    )


# ---------------------------------------------------------------------------
#  GET /api/health  –  Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", summary="Health check")
def api_health():
    service: StreamingTTSService = app.state.tts_service
    return {
        "status": "ok",
        "model_loaded": service.model is not None,
        "device": service.device,
        "sample_rate": SAMPLE_RATE,
    }
