#!/usr/bin/env python3
"""Minimal client: ask LFM3-Realtime a question and save the spoken answer.

Usage: lfm3-realtime-client [--url ws://host:port/] [--system "..."] --ask "..." --out answer.wav
Requires: ffmpeg on PATH (for mp3 -> pcm16 conversion).
"""
import argparse, asyncio, base64, json, os, subprocess, tempfile, time, sys
import numpy as np, soundfile as sf
import websockets


async def run(url: str, question: str, system: str, out: str):
    # synthesize the ask locally as PCM16 @16k via a silent-free pass:
    # simplest: send a spoken prompt. Without a mic, --ask is sent as text-like audio
    # only if you use the /demo client; here we just stream any --in file.
    raise SystemExit("use --in file.wav")


async def from_file(url, wav_in, system, out):
    w, sr = sf.read(wav_in)
    if w.ndim > 1:
        w = w[:, 0]
    if sr != 16000:
        from scipy.signal import resample_poly
        w = resample_poly(w, 16, sr)
    pcm = (np.clip(w, -1, 1) * 32767).astype(np.int16)
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "session.update",
                                  "session": {"instructions": system}}))
        await ws.recv()
        chunk = 8000  # 500ms
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + chunk].tobytes()).decode()}))
        t0 = time.monotonic()
        await ws.send(json.dumps({"type": "response.create"}))
        frames, first = [], None
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 60))
            if m["type"] == "response.audio.delta":
                if first is None:
                    first = time.monotonic() - t0
                frames.append(np.frombuffer(base64.b64decode(m["delta"]), dtype=np.int16))
            elif m["type"] == "response.done":
                break
            elif m["type"] == "error":
                raise RuntimeError(m["error"]["message"])
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        sf.write(out, audio, 24000)
        print(f"first audio in {first:.2f}s -> {out} ({audio.shape[0]/24000:.1f}s @ 24kHz)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8024/")
    ap.add_argument("--in", dest="wav_in", required=True)
    ap.add_argument("--system", default="Respond with interleaved text and audio.")
    ap.add_argument("--out", default="answer.wav")
    args = ap.parse_args()
    asyncio.run(from_file(args.url, args.wav_in, args.system, args.out))


if __name__ == "__main__":
    main()
