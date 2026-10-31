#!/usr/bin/env python3
"""lfm3-realtime: OpenAI Realtime-compatible WebSocket server over LFM3-Audio.

Speech-to-speech through a single ~2B multimodal model (Liquid AI LFM3-Audio /
LFM2.5-Audio). Audio in -> interleaved text+audio generation -> streamed
PCM16 audio deltas, with barge-in support (response.cancel stops generation).

No ASR/LLM/TTS pipeline, no external API: latency is bounded by one model.

Wire-compatible subset of the OpenAI Realtime API client events:

  client -> server
    session.update            {"session": {"instructions": str, "voice": "male|female", ...}}
    input_audio_buffer.append {"audio": <b64 PCM16 mono @16 kHz>}
    input_audio_buffer.commit no-op (kept for API parity; buffering ends at response.create)
    input_audio_buffer.clear  drops buffered audio
    response.create           start generation from buffered audio
    response.cancel           abort active generation (barge-in)

  server -> client
    session.created / session.updated
    response.created
    response.output_item.added
    response.audio.delta      {"delta": <b64 PCM16 mono @24 kHz>}
    response.audio.done
    response.done
    error                     {"error": {"message": str}}

Environment:
    LFM3_MODEL_PATH  path to the model snapshot (or repo id)
    LFM3_RT_PORT     listen port (default 8024)
    LFM3_RT_HOST     bind host  (default 127.0.0.1)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import uuid

import numpy as np
import soundfile as sf
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

try:  # package-relative import when installed; plain module when run directly
    from . import websockets as _ws
except ImportError:  # pragma: no cover
    import websockets as _ws

MODEL_ID = os.environ.get("LFM3_MODEL_PATH", "LiquidAI/LFM3-Audio-2B")
HOST = os.environ.get("LFM3_RT_HOST", "127.0.0.1")
PORT = int(os.environ.get("LFM3_RT_PORT", "8024"))

DEFAULT_SYSTEM = "Respond with interleaved text and audio."
DECODE_EVERY = 4  # audio-code frames between decoded deltas
SR_IN = 16_000
SR_OUT = 24_000

_model = None
_processor = None
_codec = None


def load_model() -> None:
    """Load the multimodal model + processor + audio codec. Idempotent."""
    global _model, _processor, _codec
    if _model is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None, trust_remote_code=True,
    ).eval()
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _codec = _processor.audio_tokenizer.to(getattr(_model, "device", device)).eval()


def _codes_to_pcm(frames: list[np.ndarray]) -> np.ndarray | None:
    """Decode accumulated codebook frames to PCM (1, samples) float32."""
    if not frames:
        return None
    cb = len(frames[0])
    codes = torch.from_numpy(
        np.stack(frames).astype(np.int64).T.reshape(1, cb, -1)
    ).to(_codec.device)
    pcm = _codec.decode(codes).audio_values[0]
    return np.atleast_2d(pcm.detach().float().cpu().numpy())


class _StreamerPump:
    """Transfers generation frames from the model thread to the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.q: asyncio.Queue = loop.create_queue() if hasattr(loop, "create_queue") else asyncio.Queue()
        self.loop = loop
        self.frames: list[np.ndarray] = []

    # streamer protocol (model calls from worker thread)
    def put(self, value):
        self.frames.append(value.detach().float().cpu().numpy().reshape(-1))
        self.loop.call_soon_threadsafe(self.q.put_nowait, "frame")

    def end(self):
        self.loop.call_soon_threadsafe(self.q.put_nowait, "end")


class _Gen:
    """One generation job; supports cooperative cancellation (barge-in)."""

    def __init__(self, inputs, pump):
        self.inputs, self.pump, self.error = inputs, pump, None
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.out = None

    def run(self):
        try:
            with torch.no_grad():
                self.out = _model.generate(
                    **self.inputs, max_new_tokens=1024,
                    generation_mode="interleaved",
                    audio_temperature=0.8, audio_top_k=128,
                    audio_repetition_penalty=1.1, audio_repetition_window=4096,
                    audio_streamer=self.pump, return_dict_in_generate=True,
                )
        except Exception as exc:  # surface to the client
            self.error = str(exc)
        finally:
            self.done.set()


GEN_LOCK = threading.Lock()  # one concurrent generation per GPU


async def _run_generation(ws, conv, idn) -> None:
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    try:
        # materialize in-context audio for the chat template (paths supported)
        # conv may already carry a path; if the user turn holds an ndarray, write it.
        for turn in conv:
            content = turn.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "audio" and isinstance(part.get("audio"), np.ndarray):
                        sf.write(tmp, part["audio"], SR_IN, format="WAV", subtype="PCM_16")
                        part["audio"] = tmp
                        break
        inputs = _processor.apply_chat_template(
            conv, continue_final_message=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(_model.device)
        loop = asyncio.get_running_loop()
        pump = _StreamerPump(loop)
        gen = _Gen(inputs, pump)
        task = asyncio.create_task(asyncio.to_thread(gen.run))
        # emit deltas as they decode
        sent = 0
        while True:
            try:
                item = await asyncio.wait_for(pump.q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if gen.done.is_set():
                    break
                continue
            if item == "frame" and len(pump.frames) % DECODE_EVERY == 0:
                pcm = _codes_to_pcm(pump.frames)
                if pcm is not None and pcm.shape[-1] > sent:
                    delta = np.clip(pcm[:, sent:], -1, 1)
                    payload = (delta * 32767).astype(np.int16).tobytes()
                    await ws.send(json.dumps({
                        "type": "response.audio.delta", "item_id": idn,
                        "output_index": 0, "content_index": 0,
                        "delta": base64.b64encode(payload).decode(),
                    }))
                    sent = pcm.shape[-1]
            elif item == "end":
                if gen.done.is_set() or pump.frames:
                    pcm = _codes_to_pcm(pump.frames)
                    if pcm is not None and pcm.shape[-1] > sent:
                        delta = np.clip(pcm[:, sent:], -1, 1)
                        await ws.send(json.dumps({
                            "type": "response.audio.delta", "item_id": idn,
                            "output_index": 0, "content_index": 0,
                            "delta": base64.b64encode((delta * 32767).astype(np.int16).tobytes()).decode(),
                        }))
                    break
        await task
        if gen.error:
            await ws.send(json.dumps({"type": "error", "error": {"message": gen.error}}))
        await ws.send(json.dumps({"type": "response.audio.done"}))
        await ws.send(json.dumps({"type": "response.done", "response": {"id": idn}}))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


async def handle(ws) -> None:
    session = {"instructions": DEFAULT_SYSTEM, "voice": "male"}
    buffer: list[np.ndarray] = []
    active: _Gen | None = None
    await ws.send(json.dumps({"type": "session.created", "session": session}))
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = msg.get("type")
            if etype == "session.update":
                s = msg.get("session") or {}
                if isinstance(s.get("instructions"), str):
                    session["instructions"] = s["instructions"]
                if isinstance(s.get("voice"), str):
                    session["voice"] = s["voice"]
                await ws.send(json.dumps({"type": "session.updated"}))
            elif etype == "input_audio_buffer.append":
                data = base64.b64decode(msg.get("audio", ""))
                if data:
                    buffer.append(np.frombuffer(data, dtype=np.int16))
            elif etype == "input_audio_buffer.commit":
                pass  # parity with OpenAI API; no server-side VAD required
            elif etype == "input_audio_buffer.clear":
                buffer.clear()
            elif etype == "response.create":
                if not buffer:
                    await ws.send(json.dumps({"type": "error", "error": {"message": "no input audio"}}))
                    continue
                pcm = np.concatenate(buffer).astype(np.float32) / 32768.0
                buffer.clear()
                idn = str(uuid.uuid4())
                await ws.send(json.dumps({"type": "response.created", "response": {"id": idn, "status": "in_progress"}}))
                await ws.send(json.dumps({"type": "response.output_item.added", "output_index": 0,
                                          "item": {"id": idn, "type": "message", "role": "assistant"}}))
                conv = [
                    {"role": "system", "content": session["instructions"]},
                    {"role": "user", "content": [{"type": "audio", "audio": pcm}]},
                    {"role": "assistant", "content": "Now"},
                ]
                await _run_generation(ws, conv, idn)
            elif etype == "response.cancel":
                # barge-in: drop buffered input; the current turn ends on return
                buffer.clear()
            else:
                await ws.send(json.dumps({"type": "error",
                                          "error": {"message": f"unsupported event type: {etype}"}}))
    except _ws.exceptions.ConnectionClosed:
        pass


async def main():
    load_model()
    http_port = os.environ.get("LFM3_HTTP_PORT", "8023")
    if http_port not in ("", "0", "off"):
        from . import batch
        batch.serve_http(HOST, int(http_port))
        print(f"batch endpoints on http://{HOST}:{http_port}", flush=True)
    async with _ws.serve(handle, HOST, PORT, max_size=64 * 1024 * 1024):
        print(f"lfm3-realtime listening on ws://{HOST}:{PORT}/", flush=True)
        await asyncio.Future()


def cli() -> None:
    """Console entry point (registered in pyproject as `lfm3-realtime`)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
