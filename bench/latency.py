#!/usr/bin/env python3
"""Realtime smoke test: stream a wav through lfm3-realtime as client events."""
import asyncio, base64, json, sys, time
import numpy as np, soundfile as sf
import websockets

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/asr_test.wav"
URL = "ws://127.0.0.1:8024/"

async def main():
    w, sr = sf.read(AUDIO)
    if w.ndim > 1:
        w = w[:, 0]
    # resample to 16k int16
    from scipy.signal import resample_poly
    w16 = resample_poly(w, 16, sr) if sr != 16000 else w
    pcm = (np.clip(w16, -1, 1) * 32767).astype(np.int16)
    chunk = int(16000 * 0.5)  # 500ms
    async with websockets.connect(URL, max_size=64 * 1024 * 1024) as ws:
        await ws.recv()  # session.created
        t0 = time.monotonic()
        first = None
        deltas = []
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + chunk].tobytes()).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                break
            m = json.loads(raw)
            if m["type"] == "response.audio.delta":
                t = time.monotonic() - t0
                if first is None:
                    first = t
                b = np.frombuffer(base64.b64decode(m["delta"]), dtype=np.int16)
                deltas.append(b)
            elif m["type"] == "response.done":
                break
            elif m["type"] == "error":
                print("ERROR:", m["error"]); break
        if deltas:
            out = np.concatenate(deltas)
            sf.write("/tmp/rt_out.wav", out.astype(np.float32) / 32768.0, 24000)
            print(f"OK first_delta={first:.3f}s total={time.monotonic()-t0:.2f}s audio={out.shape[0]/24000:.1f}s")
        else:
            print("NO AUDIO DELTAS")

asyncio.run(main())
