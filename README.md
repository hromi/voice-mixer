# voicelab

A communal voice-cloning toolkit built on [OpenVoice](https://github.com/myshell-ai/OpenVoice) V2.
It exposes the same core engine through a CLI, an HTTP API, and a Gradio web
app (with in-browser recording), and adds a storage layer that human
recordings and every cloned result flow through.

## Layout

```
voices/<Speaker>/            human reference recordings, one subfolder per speaker
synthetic/<Speaker[+Speaker2]>/   cached cloned audio + metadata sidecars
checkpoints/                 OpenVoice model weights (downloaded, not committed)
voicelab/                    the Python package (storage, embeddings, engine, cli, api, webapp)
```

Adding a recording for `SolarPunk0` creates `voices/SolarPunk0/<file>.wav`
(plus a small `.json` sidecar with upload metadata). Cloning text in that
voice writes to `synthetic/SolarPunk0/<id>.wav` with a sibling
`<id>.json` recording the **prompt text and every parameter** needed to
reproduce that exact clone: language, speed, tau (conversion strength),
resolved base speaker, and which source recordings (with checksums) fed
the tone-color embedding. Identical requests are served from this cache
instead of re-running the model — see `storage.compute_cache_key` /
`storage.find_cached`.

### Communal cloning (voice crossover)

A "speaker" for cloning purposes doesn't have to be one person. Pass a
weighted mix of multiple speakers and their tone-color embeddings are
averaged (optionally restricted to specific recordings per speaker) before
conversion — a genuine crossover in embedding space between two or more
human voices:

```
voicelab clone "Hello, this is a blended voice." --mix "SolarPunk0:0.6,JaneDoe:0.4"
```

The result lands in `synthetic/JaneDoe+SolarPunk0/` (mix folders are
named by sorting the participating speakers), with the metadata recording
each speaker's normalized weight and exact source files.

### Languages / TTS backends

Cloning always works the same way — a base TTS engine speaks the text in
some stock voice, then OpenVoice's tone-color conversion transfers it to
the target speaker(s). Which engine generates that base utterance depends
on the language:

| Backend | Languages | Notes |
|---|---|---|
| MeloTTS | EN, EN_NEWEST, ES, FR, ZH, JP, KR | `--speed` applies |
| Qwen3-TTS | DE, RU, PT, IT | MeloTTS has no voice for these; needs a CUDA GPU; `--speed` has no effect |

`--base-speaker` picks the base voice within whichever backend is in play
(a MeloTTS speaker id like `EN-Default`, or one of Qwen3-TTS's 9 preset
names like `Ryan`/`Vivian`/`Sohee` — left blank, it auto-picks one).

For the Qwen backend specifically, `--qwen-clone-method` picks *how* your
speakers' voices actually reach the output:

- `openvoice` (default) — Qwen speaks a preset voice, then OpenVoice's
  tone-color conversion retargets it to your mix. Same mechanism as
  MeloTTS, works identically for any backend.
- `native` — skips OpenVoice entirely. Qwen3-TTS's separate `-Base`
  checkpoint extracts an x-vector straight from your speakers' recordings
  (`model.extract_speaker_embedding(...)`) and generates already
  conditioned on the blended embedding, in one native pass. Implies
  `--backend qwen`; downloads another ~4GB checkpoint on first use.
  `--tau` and `--base-speaker` have no effect here (nothing gets
  converted, no preset voice is used).

Both methods support the same weighted communal mixing and per-file
pinpointing — mixing happens on the embedding side either way, it's just
OpenVoice's tone-color space vs. Qwen's own x-vector space.

Qwen3-TTS (`qwen-tts` on PyPI) uses Python 3.10+ syntax that this
project's main interpreter may not support, so it doesn't live in the main
environment at all — it runs as a small subprocess in its own venv
(`.venv-qwen/`, set up by `scripts/setup_qwen_venv.sh`) that `engine.py`
talks to over stdin/stdout. This also needs a CUDA GPU; there's no CPU
path for Qwen3-TTS.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt        # CLI / API / web app (no cloning yet)
pip install -r requirements-ml.txt     # + torch, OpenVoice, MeloTTS (needed to actually clone)
python -m unidic download              # MeloTTS's tokenizer data
scripts/download_checkpoints.sh        # OpenVoice V2 weights -> ./checkpoints
scripts/setup_qwen_venv.sh             # optional: DE/RU/PT/IT via Qwen3-TTS (needs a CUDA GPU + Python 3.10+)
```

The CLI, API, and web app all start up fine with just `requirements.txt`
(useful for browsing/uploading); actually synthesizing audio needs
`requirements-ml.txt` plus the downloaded checkpoints.
`scripts/setup_qwen_venv.sh` is only needed for
German/Russian/Portuguese/Italian — its checkpoint (~4GB) downloads
automatically from Hugging Face the first time you clone in one of those
languages, no separate download step required.

## CLI

```bash
voicelab voices add SolarPunk0 sample1.wav sample2.wav
voicelab voices list
voicelab clone "Hello there" --mix SolarPunk0
voicelab clone "Hello there" --mix "SolarPunk0:0.6,JaneDoe:0.4" --language EN --speed 1.1
voicelab clone "Hallo, wie geht es dir?" --mix SolarPunk0 --language DE   # via Qwen3-TTS
voicelab clone "Hallo, wie geht es dir?" --mix SolarPunk0 --qwen-clone-method native  # Qwen's own x-vector cloning
voicelab synthetic list
voicelab serve api   # http://127.0.0.1:8000
voicelab serve web   # http://127.0.0.1:7860
```

## HTTP API

Once running (`voicelab serve api`), interactive docs are at `/docs`. Key
endpoints:

- `GET /speakers` — list speakers and recording counts
- `POST /speakers/{speaker}/recordings` — upload a recording (multipart)
- `POST /synthesize` — `{"text": "...", "mix": [{"speaker": "SolarPunk0", "weight": 0.6}, {"speaker": "JaneDoe", "weight": 0.4}]}`
- `GET /synthetic/{folder}/{id}/audio` — fetch a cached clone's audio

## Web app

`voicelab serve web` opens a Gradio app with three tabs: record or upload a
voice straight from the browser microphone, clone/mix text into speech
with adjustable weights and parameters, and browse the recording/synthetic
library with playback and full metadata.

## Tests

```bash
pip install pytest
pytest
```

Tests cover the storage and embedding-blend logic with lightweight
fixtures and do not require torch/OpenVoice/checkpoints to be installed.
