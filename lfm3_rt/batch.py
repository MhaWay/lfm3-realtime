"""HTTP batch endpoints (TTS/ASR/understand/translate) alongside the realtime WebSocket.

Same model instance, same generation lock as the realtime path:
  POST /tts_to_audio  {"text": ...}   -> wav 24kHz (interleaved? no: sequential TTS)
  POST /tts_raw       {"text": ...}   -> wav, same as tts_to_audio
  POST /asr           {"audio_path"|"audio_b64", "language": "italian"|""} -> {"text"}
  POST /understand    {"audio_path"|"audio_b64", "question": ...}          -> {"text"}
  POST /translate_speech {"audio_path"|"audio_b64", "source": ..., "target": ...} -> {"text"}
  POST /translate_text   {"text": ..., "source": ..., "target": ...}       -> {"text"}
  GET  /health        -> {"ok": true, "device": ...}

Base64 audio is decoded to a temp file because the chat template accepts paths.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
import torch

from . import server as _srv
from .server import GEN_LOCK, SR_OUT, load_model  # noqa: F401


def _model():
    return _srv._model


def _processor():
    return _srv._processor

VOICE = os.environ.get("LFM3_VOICE", "male")
GEN = threading.Lock()  # module-level; shared semantics with realtime lock


def _decode_audio(body: dict) -> str:
    """Materialize audio to a temp wav path and return it."""
    p = body.get("audio_path")
    if p:
        return p
    b64 = body.get("audio_b64")
    if not b64:
        raise ValueError("no audio provided: need audio_path or audio_b64")
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(b64))
    return tmp


def synthesize(text: str) -> np.ndarray:
    """Sequential-mode Italian TTS -> float PCM @24kHz."""
    conv = [
        {"role": "system", "content": f"Perform TTS in Italian using a {VOICE} voice."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "Now"},
    ]
    inputs = _processor().apply_chat_template(
        conv, continue_final_message=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(_model().device)
    with torch.no_grad(), GEN_LOCK:
        out = _model().generate(
            **inputs, max_new_tokens=1536, generation_mode="audio",
            audio_temperature=0.8, audio_top_k=100,
            audio_repetition_penalty=1.2, audio_repetition_window=64,
            return_dict_in_generate=True)
    k = int(out.audio_lengths[0].item())
    if k == 0:
        raise RuntimeError("no audio generated")
    pcm = _processor().audio_tokenizer.decode(out.audio_codes[:1, :, :k]).audio_values[0]
    w = pcm.detach().float().cpu().numpy()
    return w / max(1e-6, np.max(np.abs(w))) * 0.9


def transcribe(path: str, language: str) -> str:
    sysmsg = f"Perform ASR in {language}." if language else "Perform ASR."
    return _gen_text([{"role": "system", "content": sysmsg},
                      {"role": "user", "content": [{"type": "audio", "audio": path}]}])


def understand(path: str, question: str) -> str:
    return _gen_text([
        {"role": "system", "content": "Answer questions about the provided audio. Base every answer only on what can be heard."},
        {"role": "user", "content": [{"type": "audio", "audio": path},
                                      {"type": "text", "text": question}]},
    ])


def _gen_text(conv, max_new=512) -> str:
    inputs = _processor().apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(_model().device)
    with torch.no_grad(), GEN_LOCK:
        out = _model().generate(**inputs, max_new_tokens=max_new, generation_mode="text")
    return _processor().decode(out.sequences[0, inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


class BatchHandler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _wav(self, pcm):
        buf = io.BytesIO()
        sf.write(buf, np.clip(pcm, -1, 1) * 0.9, SR_OUT, format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            self._json(200, {"ok": True, "device": str(_model().device), "mode": "unified"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            p = self.path.split("?")[0]
            if p in ("/tts_to_audio", "/tts_raw"):
                text = (body.get("text") or "").strip()
                if not text:
                    raise ValueError("missing text")
                self._wav(synthesize(text))
            elif p == "/asr":
                self._json(200, {"text": transcribe(_decode_audio(body), (body.get("language") or "").strip())})
            elif p == "/understand":
                self._json(200, {"text": understand(_decode_audio(body), body.get("question", "Describe this audio."))})
            elif p == "/translate_speech":
                path = _decode_audio(body)
                self._json(200, {"text": _gen_text([
                    {"role": "system", "content": f"Translate the following speech from {body.get('source', 'italian')} to {body.get('target', 'english')} in text form."},
                    {"role": "user", "content": [{"type": "audio", "audio": path}]}])})
            elif p == "/translate_text":
                self._json(200, {"text": _gen_text([
                    {"role": "system", "content": f"Translate the following {body.get('source', 'italian')} text into {body.get('target', 'english')}."},
                    {"role": "user", "content": body.get("text", "")}])})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, *args):
        pass


def serve_http(host: str = "127.0.0.1", port: int = 8023) -> ThreadingHTTPServer:
    load_model()
    srv = ThreadingHTTPServer((host, port), BatchHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
