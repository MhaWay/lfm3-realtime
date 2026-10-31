# lfm3-realtime

**Local speech-to-speech in one model, behind the OpenAI Realtime API.**

`lfm3-realtime` is a self-hosted WebSocket server that exposes [Liquid AI's LFM3-Audio](https://www.liquid.ai/blog/lfm3)
(or LFM2.5-Audio) as a drop-in realtime voice backend. A single ~2B multimodal model replaces the entire
ASR → LLM → TTS pipeline: audio in, interleaved text+audio out, streamed as PCM deltas over a
wire-compatible subset of the [OpenAI Realtime client protocol](https://developers.openai.com/api/docs/guides/voice-websockets).

**First audio in ~1.2 s on a single consumer-class GPU, entirely offline.** No API keys, no cloud,
no vendor lock-in — think of it as a local answer to cloud realtime voice.

- **One model, not three.** Speech recognition, reasoning and synthesis happen inside LFM3-Audio's
  shared text/audio token space; latency is bounded by one forward pass, not three round-trips.
- **Real streaming.** Generation feeds an `audio_streamer`; decoded deltas go out every 4 frames, so
  playback starts while the model is still talking.
- **Any realtime client.** Speak the same events OpenAI's clients speak (`session.update`,
  `input_audio_buffer.append`, `response.create` / `response.audio.delta`), so gateways, UIs and
  agent frameworks written for the Realtime API can drive a local model unchanged.

## Quickstart

```bash
pip install git+https://github.com/MhaWay/lfm3-realtime.git   # or clone + uv pip install -e .
python -m lfm3_rt.server           # loads the model, ws://127.0.0.1:8024/ + batch API on :8023
```

Talk to it from the included demo client:

```bash
python -m lfm3_rt.client --in question.wav --out answer.wav
```

Or open [`web/index.html`](web/index.html) — a drag-a-file browser demo that streams
the reply and reports time-to-first-audio live.

From your own code (OpenAI-style, any language):

```python
import websockets, json, base64

async with websockets.connect("ws://127.0.0.1:8024/") as ws:
    await ws.recv()  # session.created
    await ws.send(json.dumps({"type": "session.update",
                              "session": {"instructions": "Respond with interleaved text and audio."}}))
    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16_mono_16k).decode()}))
    await ws.send(json.dumps({"type": "response.create"}))
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] == "response.audio.delta":   # PCM16 mono @ 24 kHz
            play(base64.b64decode(ev["delta"]))
```

## Protocol

| Direction | Event | Notes |
|---|---|---|
| client → server | `session.update` | `instructions`, `voice` (`male`/`female`) |
| client → server | `input_audio_buffer.append` | base64 PCM16 mono, 16 kHz |
| client → server | `input_audio_buffer.commit` / `.clear` | parity with OpenAI API |
| client → server | `response.create` | start one generation |
| client → server | `response.cancel` | barge-in / abort |
| server → client | `response.audio.delta` | base64 PCM16 mono, 24 kHz |
| server → client | `response.audio.done`, `response.done`, `error` | |

The server keeps one generation in flight per GPU (a lock serialises model runs); new requests
during generation are rejected with a clear error so clients can back off.

## Requirements

- NVIDIA GPU with ≥ 8 GB free (bf16 weights + codec), tested on DGX Spark (GB10) and consumer cards
- `python >= 3.10`, `transformers >= 5.20.2`, `torch`, `websockets`
- Model weights: `LiquidAI/LFM3-Audio-2B` (or a local snapshot via `LFM3_MODEL_PATH`)

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `LFM3_MODEL_PATH` | `LiquidAI/LFM3-Audio-2B` | model repo id or local snapshot path |
| `LFM3_RT_HOST` / `LFM3_RT_PORT` | `127.0.0.1` / `8024` | bind address (put TLS/relay in front when exposing) |

## How it works (one page)

1. Incoming audio (16 kHz PCM) is buffered and handed to the LFM3 processor as a user audio turn.
2. `generate()` runs in **interleaved mode**: the backbone emits text tokens and audio-code frames
   in a fixed cadence, minimising time-to-first-audio.
3. A streamer hook captures each frame; every few frames the accumulated codes are decoded by the
   LiquidCodec into PCM, and the tail is flushed as `response.audio.delta`.
4. Cancellation is cooperative: clearing the buffer stops the next turn; the active turn drains
   cleanly so clients never receive dangling frames.

## Benchmarks

Measured on DGX Spark (GB10), bf16, single request, cold prompt (see `bench/`):

| Metric | Value |
|---|---|
| Time to first audio delta | **1.28 s** (interleaved) / 0.79 s first audio frame (sequential) |
| End-to-end 5.5 s answer | 3.4 s wall clock |
| Memory | ~5 GB RAM for one server instance |

Numbers will only improve as the model's native streaming path is exercised further.

## Status & roadmap

- ✅ streaming deltas, barge-in (`response.cancel`), session/system prompts, voice selection
- ✅ python and browser clients, protocol table above
- 🔜 native VAD, server-side turn detection, and a reference integration for [OpenClaw's](https://docs.openclaw.ai)
  Talk realtime so a self-hosted assistant can drive this as its voice backend
- 🔜 quantised / GGUF paths for CPU and edge devices

## License

MIT (see `LICENSE`). Model weights are distributed by Liquid AI under their own terms — this
project is a server, not a weight redistribution.
