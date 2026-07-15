# voicelab

A communal voice-cloning toolkit. The primary engine is **Qwen3-TTS's
native speaker-embedding cloning**: extract an x-vector straight from your
speakers' recordings, blend them, and generate speech directly conditioned
on that blend — one model, one pass, no separate voice-conversion step.
[OpenVoice](https://github.com/myshell-ai/OpenVoice) V2 tone-color
conversion (with MeloTTS or Qwen3-TTS providing the stock base voice) is
available as an **alternative pipeline** — useful for speed control, or to
compare quality against the native path.

It exposes the same core engine through a CLI, an HTTP API, and a Gradio
web app (with in-browser recording), and adds a storage layer that human
recordings and every cloned result flow through.

## Layout

```
voices/<Speaker>/            human reference recordings, one subfolder per speaker
synthetic/<Speaker[+Speaker2]>/   cached cloned audio + metadata sidecars
checkpoints/                 OpenVoice model weights (downloaded, not committed; only needed for the alternative pipeline)
voicelab/                    the Python package (storage, embeddings, engine, cli, api, webapp)
```

Adding a recording for `SolarPunk0` creates `voices/SolarPunk0/<file>.wav`
(plus a small `.json` sidecar with upload metadata, including an optional
note/transcript). Cloning text in that voice writes to
`synthetic/SolarPunk0/<id>.wav` with a sibling `<id>.json` recording the
**prompt text and every parameter** needed to reproduce that exact clone —
which pipeline was used, language, and which source recordings (with
checksums) fed the embedding. Identical requests are served from this
cache instead of re-running the model — see `storage.compute_cache_key` /
`storage.find_cached`.

## How cloning works

Every clone starts from a weighted mix of one or more known speakers'
recordings. What differs between the two pipelines is *how* that mix
turns into audio:

**Qwen3-TTS native (recommended, default in the web app)** — the
`Qwen3-TTS-12Hz-1.7B-Base` checkpoint extracts a speaker x-vector from
each recording (`model.extract_speaker_embedding(...)`), averages within
each speaker and blends across speakers by weight, then generates the
target text in one pass directly conditioned on that blended embedding.
No intermediate "stock voice" step, no separate conversion model. Covers
10 languages: English, Spanish, French, Chinese, Japanese, Korean,
German, Russian, Portuguese, Italian. Needs a CUDA GPU.

**OpenVoice tone-color conversion (alternative)** — a base TTS engine
(MeloTTS, or Qwen3-TTS speaking one of its 9 preset voices) speaks the
text in some stock voice first, then OpenVoice's `ToneColorConverter`
retargets that stock voice's timbre to your blended speaker mix as a
second pass. This is the only path that supports `--speed` control
(MeloTTS only — Qwen has no speed parameter either way), and doesn't need
the extra ~4GB `-Base` checkpoint. Pick it with `--qwen-clone-method
openvoice` (Qwen backend) or by using the MeloTTS backend directly.

Both pipelines support the same weighted communal mixing and per-file
pinpointing — blending just happens in a different embedding space
(Qwen's x-vectors vs. OpenVoice's tone-color embeddings).

### Communal cloning (voice crossover)

A "speaker" for cloning purposes doesn't have to be one person. Pass a
weighted mix of multiple speakers and their embeddings are blended
(optionally restricted to specific recordings per speaker) before
generation — a genuine crossover between two or more human voices:

```
voicelab clone "Hello, this is a blended voice." --mix "SolarPunk0:0.6,JaneDoe:0.4" --qwen-clone-method native
```

The result lands in `synthetic/JaneDoe+SolarPunk0/` (mix folders are
named by sorting the participating speakers), with the metadata recording
each speaker's normalized weight and exact source files.

### Languages / pipelines

| Pipeline | Languages | Notes |
|---|---|---|
| **Qwen3-TTS native** (recommended) | EN, ES, FR, ZH, JP, KR, DE, RU, PT, IT | Needs a CUDA GPU; no `--speed` control; `--tau`/`--base-speaker` don't apply (no preset voice, nothing gets converted) |
| OpenVoice + Qwen3-TTS preset (alternative) | same 10, via Qwen's preset voices | `--qwen-clone-method openvoice` |
| OpenVoice + MeloTTS (alternative) | EN, EN_NEWEST, ES, FR, ZH, JP, KR | `--backend melo`; `--speed` applies; no CUDA GPU strictly required |

`--backend` picks the base-TTS engine (`melo` or `qwen`, auto-selected by
language if omitted — Qwen only kicks in automatically for DE/RU/PT/IT,
which MeloTTS can't do at all). `--qwen-clone-method` then picks *how*
the Qwen backend turns your mix into audio — `native` (recommended) or
`openvoice`; requesting `native` implies `--backend qwen`. `--base-speaker`
only matters for the OpenVoice pipeline: a MeloTTS speaker id like
`EN-Default`, or one of Qwen3-TTS's 9 preset names like `Ryan`/`Vivian`/
`Sohee` — left blank, it auto-picks one.

Qwen3-TTS (`qwen-tts` on PyPI) uses Python 3.10+ syntax that this
project's main interpreter may not support, so it doesn't live in the main
environment at all — it runs as a small subprocess in its own venv
(`.venv-qwen/`, set up by `scripts/setup_qwen_venv.sh`) that `engine.py`
talks to over stdin/stdout. Both the `-Base` (native cloning) and
`-CustomVoice` (preset voices, for the OpenVoice pipeline) checkpoints
live in that same venv/process and download automatically from Hugging
Face the first time each is actually used.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt        # CLI / API / web app (no cloning yet)
scripts/setup_qwen_venv.sh             # Qwen3-TTS (native cloning + the preset-voice option) — needs a CUDA GPU + Python 3.10+
```

That's enough for the recommended (Qwen3-TTS native) path — its
checkpoints (~4GB each for `-Base` and `-CustomVoice`) download
automatically on first use, no separate download step.

The OpenVoice/MeloTTS pipeline is optional, only needed if you want the
alternative tone-color-conversion path or MeloTTS's `--speed` control:

```bash
pip install -r requirements-ml.txt     # + torch, OpenVoice, MeloTTS
python -m unidic download              # MeloTTS's tokenizer data
scripts/download_checkpoints.sh        # OpenVoice V2 weights -> ./checkpoints
```

The CLI, API, and web app all start up fine with just `requirements.txt`
(useful for browsing/uploading); actually synthesizing audio needs at
least one of the two setups above.

## CLI

```bash
voicelab voices add SolarPunk0 sample1.wav sample2.wav
voicelab voices list
voicelab clone "Hello there" --mix SolarPunk0 --qwen-clone-method native      # recommended
voicelab clone "Hallo, wie geht es dir?" --mix SolarPunk0 --qwen-clone-method native
voicelab clone "Hello there" --mix "SolarPunk0:0.6,JaneDoe:0.4" --qwen-clone-method native
voicelab clone "Hello there" --mix SolarPunk0 --backend melo --speed 1.1      # alternative pipeline
voicelab synthetic list
voicelab serve api   # http://127.0.0.1:8000
voicelab serve web   # http://127.0.0.1:7860
```

## HTTP API

Once running (`voicelab serve api`), interactive docs are at `/docs`. Key
endpoints:

- `GET /speakers` — list speakers and recording counts
- `POST /speakers/{speaker}/recordings` — upload a recording (multipart)
- `POST /synthesize` — `{"text": "...", "mix": [{"speaker": "SolarPunk0", "weight": 0.6}, {"speaker": "JaneDoe", "weight": 0.4}], "qwen_clone_method": "native"}`
- `GET /synthetic/{folder}/{id}/audio` — fetch a cached clone's audio

## Web app

`voicelab serve web` opens a Gradio app (behind a login wall — set
`VOICELAB_WEB_PASSWORD`, see `voicelab/config.py`) with three tabs: record
or upload a voice straight from the browser microphone, clone/mix text
into speech with per-speaker weight sliders that auto-rebalance, and
browse the recording/synthetic library with playback and full metadata.
It defaults to the recommended Qwen3-TTS native pipeline; the OpenVoice
alternative is one radio button away.

## Tests

```bash
pip install pytest
pytest
```

Tests cover the storage and embedding-blend logic with lightweight
fixtures and do not require torch/OpenVoice/Qwen3-TTS/checkpoints to be
installed.
