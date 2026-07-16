"""Thin wrapper around OpenVoice V2 (MeloTTS base speaker + ToneColorConverter)
that adds content-addressed caching and full-reproducibility metadata.

Heavy ML dependencies (torch, openvoice, melo) are imported lazily inside
methods so that the CLI, API, and web app can start up (e.g. `--help`,
listing speakers) without them installed. They're only required once an
actual synthesis is requested.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from . import chorus, config, embeddings, storage
from .embeddings import MixEntry

_ENGINE_VERSION = "2"

MixSpec = list[Union[MixEntry, dict]]


class QwenWorkerError(RuntimeError):
    pass


class _QwenWorkerProcess:
    """Talks to qwen_worker.py running in its own Python 3.10+ venv (see
    config.QWEN_VENV_PYTHON) over newline-delimited JSON on stdin/stdout.
    Kept alive for the engine's lifetime so the ~GB model only loads once.
    """

    def __init__(self, custom_voice_model_id: str, base_model_id: str, device: str):
        if not config.QWEN_VENV_PYTHON.exists():
            raise CheckpointsNotFound(
                f"Qwen3-TTS venv not found at {config.QWEN_VENV_PYTHON}. "
                "Run scripts/setup_qwen_venv.sh first (needs Python 3.10+ and a CUDA GPU)."
            )
        worker_script = Path(__file__).with_name("qwen_worker.py")
        self._proc = subprocess.Popen(
            [str(config.QWEN_VENV_PYTHON), str(worker_script), custom_voice_model_id, base_model_id, device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._supported_speakers: Optional[list[str]] = None
        ready = self._read_response()
        if not ready.get("ok"):
            raise QwenWorkerError(f"Qwen3-TTS worker failed to start: {ready}")

    def _read_response(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read()
            raise QwenWorkerError(f"Qwen3-TTS worker exited unexpectedly.\n{stderr}")
        return json.loads(line)

    def _request(self, payload: dict) -> dict:
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        resp = self._read_response()
        if not resp.get("ok"):
            raise QwenWorkerError(f"Qwen3-TTS error: {resp.get('error')}")
        return resp

    def supported_speakers(self) -> list[str]:
        if self._supported_speakers is None:
            self._supported_speakers = self._request({"cmd": "supported_speakers"})["result"]
        return self._supported_speakers

    def generate_preset(self, text: str, language: str, speaker: str, out_path: Path) -> None:
        self._request({
            "cmd": "generate_preset", "text": text, "language": language,
            "speaker": speaker, "out_path": str(out_path),
        })

    def generate_native_clone(self, text: str, language: str, mix: list[dict], out_path: Path) -> None:
        """`mix` is [{"files": [absolute paths], "weight": float}, ...] — one
        entry per speaker, mirroring embeddings.communal_embedding's shape."""
        self._request({
            "cmd": "generate_native_clone", "text": text, "language": language,
            "mix": mix, "out_path": str(out_path),
        })

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass  # nothing more we can do; avoid blocking shutdown on it


def _stub_unused_cloud_backends():
    """MeloTTS's import chain drags in two cloud-storage SDKs it never
    actually needs for our use case, and both are broken to import for
    real in this environment:

    - transformers -> accelerate has an *optional*, guarded `import boto3`
      (for an unrelated SageMaker CLI helper). boto3 being installed makes
      accelerate execute that import for real, which loads this
      environment's old botocore — incompatible with the modern urllib3
      that gradio/huggingface_hub require (an unwinnable version conflict,
      since we don't use boto3 at all).
    - MeloTTS -> cached_path unconditionally imports its S3/GCS/R2 backend
      modules (cached_path/schemes/{s3,gs,r2}.py) even though we only ever
      resolve local paths or hf:// URLs through it. s3.py/r2.py hit the
      same boto3/botocore problem; gs.py needs google-cloud-storage, which
      isn't installed at all.

    Both are dead code for us — every use of these classes lives inside
    method bodies that only run for gs://, s3://, r2:// resources we never
    pass in. Pre-seed sys.modules with minimal stand-ins so the *import*
    statements succeed without pulling in (or fighting the broken version
    of) either real SDK.
    """
    import importlib.machinery
    import sys
    import types

    def _stub(name: str, **attrs):
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__path__ = []
        for attr, value in attrs.items():
            setattr(mod, attr, value)
        sys.modules[name] = mod
        if "." in name:
            parent = sys.modules.get(name.rsplit(".", 1)[0])
            if parent is not None:
                setattr(parent, name.rsplit(".", 1)[1], mod)
        return mod

    _stub("boto3")
    _stub("boto3.session", Session=type("Session", (), {}))
    _stub("boto3.dynamodb")
    _stub("botocore", UNSIGNED=object())
    _stub(
        "botocore.exceptions",
        ClientError=type("ClientError", (Exception,), {}),
        HTTPClientError=type("HTTPClientError", (Exception,), {}),
        ConnectionError=type("ConnectionError", (Exception,), {}),
    )
    _stub("botocore.config", Config=type("Config", (), {}))
    _stub("botocore.client", Config=type("Config", (), {}))

    _stub("google")
    _stub("google.cloud")
    _stub("google.cloud.storage", Blob=type("Blob", (), {}), Client=type("Client", (), {}))
    _stub("google.cloud.storage.blob", Blob=type("Blob", (), {}))
    _stub("google.cloud.storage.retry", DEFAULT_RETRY=None)
    _stub("google.api_core")
    _stub("google.api_core.exceptions", NotFound=type("NotFound", (Exception,), {}))
    _stub("google.auth")
    _stub("google.auth.exceptions", DefaultCredentialsError=type("DefaultCredentialsError", (Exception,), {}))


class CheckpointsNotFound(RuntimeError):
    pass


@dataclass
class SynthesisResult:
    item: storage.SyntheticItem
    cached: bool


class OpenVoiceEngine:
    """Stateful wrapper: caches the loaded tone-color converter and one
    MeloTTS base model per language across calls, since both are expensive
    to construct."""

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._converter = None
        self._tts_models: dict[str, Any] = {}
        self._qwen_worker: Optional[_QwenWorkerProcess] = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def converter(self):
        if self._converter is None:
            self._converter = self._load_converter()
        return self._converter

    def _load_converter(self):
        from openvoice.api import ToneColorConverter

        config_path = config.CONVERTER_DIR / "config.json"
        ckpt_path = config.CONVERTER_DIR / "checkpoint.pth"
        if not config_path.exists() or not ckpt_path.exists():
            raise CheckpointsNotFound(
                f"OpenVoice converter checkpoints not found under {config.CONVERTER_DIR}. "
                "Run scripts/download_checkpoints.sh first."
            )
        converter = ToneColorConverter(str(config_path), device=self.device)
        converter.load_ckpt(str(ckpt_path))
        return converter

    def _tts_model(self, language: str):
        if language not in self._tts_models:
            _stub_unused_cloud_backends()
            from melo.api import TTS
            self._tts_models[language] = TTS(language=language, device=self.device)
        return self._tts_models[language]

    @property
    def qwen_worker(self) -> _QwenWorkerProcess:
        if self._qwen_worker is None:
            self._qwen_worker = _QwenWorkerProcess(config.QWEN_MODEL_ID, config.QWEN_BASE_MODEL_ID, self.device)
        return self._qwen_worker

    def close(self) -> None:
        if self._qwen_worker is not None:
            self._qwen_worker.close()
            self._qwen_worker = None

    def _generate_base_audio_melo(self, text: str, language: str, speed: float,
                                   base_speaker_key: Optional[str], out_path: Path) -> tuple[str, Any]:
        """Returns (resolved_base_speaker_key, source_se) after writing base audio to out_path."""
        import torch

        model = self._tts_model(language)
        resolved_key = base_speaker_key or next(iter(model.hps.data.spk2id.keys()))
        if resolved_key not in model.hps.data.spk2id:
            raise ValueError(
                f"unknown base speaker {resolved_key!r} for language {language!r}; "
                f"available: {sorted(model.hps.data.spk2id.keys())}"
            )
        speaker_id = model.hps.data.spk2id[resolved_key]

        se_name = resolved_key.lower().replace("_", "-")
        source_se_path = config.BASE_SPEAKER_SE_DIR / f"{se_name}.pth"
        if not source_se_path.exists():
            raise CheckpointsNotFound(f"missing base speaker embedding: {source_se_path}")
        source_se = torch.load(source_se_path, map_location=self.device)

        model.tts_to_file(text, speaker_id, str(out_path), speed=speed)
        return resolved_key, source_se

    def _generate_base_audio_qwen(self, text: str, qwen_language: str, base_speaker_key: Optional[str],
                                   out_path: Path) -> tuple[str, Any]:
        """Returns (resolved_speaker_name, source_se) after writing base audio to out_path.

        MeloTTS has no German (or Russian/Portuguese/Italian) voice, so for
        those languages Qwen3-TTS generates the base utterance instead —
        same downstream OpenVoice tone-color-conversion step as the Melo
        path. qwen-tts needs Python 3.10+ (this project runs on 3.9), so it
        actually runs as a subprocess in its own venv; see
        _QwenWorkerProcess / qwen_worker.py. Unlike Melo, OpenVoice never
        shipped a precomputed source embedding for Qwen's preset speakers,
        so it's extracted on the fly from the very clip just generated
        (same technique embeddings.py uses for human recordings) rather
        than needing a separate calibration step.
        """
        worker = self.qwen_worker
        resolved_speaker = base_speaker_key or config.QWEN_DEFAULT_SPEAKER
        supported_speakers = worker.supported_speakers()
        if supported_speakers and resolved_speaker.lower() not in {s.lower() for s in supported_speakers}:
            raise ValueError(
                f"unknown Qwen3-TTS speaker {resolved_speaker!r}; available: {sorted(supported_speakers)}"
            )

        worker.generate_preset(text, qwen_language, resolved_speaker, out_path)

        source_se = self.converter.extract_se([str(out_path)])
        return resolved_speaker, source_se

    def _generate_qwen_native_clone(self, text: str, qwen_language: str,
                                     resolved_mix: list[dict], out_path: Path) -> None:
        """Generates `text` directly in the blended target voice using
        Qwen3-TTS's own x-vector conditioning (the -Base checkpoint) — no
        OpenVoice tone-color conversion involved at all. `resolved_mix` is
        the same per-speaker files/weight structure `synthesize()` already
        computes for the cache key; this mirrors embeddings.py's
        per-speaker-average-then-weighted-blend semantics, just executed
        inside the worker's own process/venv since the embeddings never
        need to leave it.
        """
        worker = self.qwen_worker
        mix_payload = [
            {
                "files": [str(storage.speaker_dir(entry["speaker"]) / f) for f in entry["files"]],
                "weight": entry["weight"],
            }
            for entry in resolved_mix
        ]
        worker.generate_native_clone(text, qwen_language, mix_payload, out_path)

    def synthesize(
        self,
        text: str,
        mix: MixSpec,
        *,
        language: str = config.DEFAULT_LANGUAGE,
        speed: float = config.DEFAULT_SPEED,
        tau: float = config.DEFAULT_TAU,
        base_speaker_key: Optional[str] = None,
        backend: Optional[str] = None,
        qwen_clone_method: Optional[str] = None,
        force: bool = False,
    ) -> SynthesisResult:
        """Clone `text` in the blended voice described by `mix`.

        `mix` is a list of MixEntry (or equivalent dicts with keys
        speaker/weight/files) describing a communal blend: a single entry
        is a plain clone, multiple entries perform a weighted crossover of
        several speakers' tone-color embeddings. Results are cached under
        synthetic/<speakers>/ keyed by every parameter that affects the
        output, so identical requests are served instantly and never
        recompute the (expensive) model forward passes.

        `backend` picks which engine generates the base utterance: "melo",
        "qwen", or None/"auto" (default) to pick automatically — Qwen3-TTS
        only for languages MeloTTS has no voice for at all (DE/RU/PT/IT).
        Several languages (EN/ES/FR/ZH/JP/KR) are covered by both; forcing
        "qwen" for those is valid too, e.g. to compare quality.

        `qwen_clone_method` only matters when the Qwen backend is in play:
        "openvoice" (default) — Qwen speaks in a preset voice, then
        OpenVoice's tone-color conversion retargets it to `mix`, same
        mechanism as the Melo path. "native" — skips OpenVoice entirely;
        Qwen3-TTS's own -Base checkpoint extracts speaker x-vectors
        straight from `mix`'s recordings and generates already in that
        blended voice. Requesting "native" implies backend="qwen".
        """
        mix_entries = [m if isinstance(m, MixEntry) else MixEntry(**m) for m in mix]
        folder_name = storage.mix_folder_name([m.speaker for m in mix_entries])

        resolved_mix = []
        for m in mix_entries:
            files = sorted(m.files) if m.files else sorted(p.name for p in storage.list_recordings(m.speaker))
            if not files:
                raise ValueError(f"speaker {m.speaker!r} has no recordings in {storage.speaker_dir(m.speaker)}")
            checksums = {f: storage.checksum_file(storage.speaker_dir(m.speaker) / f) for f in files}
            resolved_mix.append({
                "speaker": m.speaker, "weight": m.weight, "files": files, "checksums": checksums,
            })

        requested_backend = (backend or "auto").strip().lower()
        if requested_backend not in ("auto", "melo", "qwen"):
            raise ValueError(f"unknown backend {backend!r}; choose 'melo', 'qwen', or leave blank for auto")

        requested_qwen_clone_method = (qwen_clone_method or "openvoice").strip().lower()
        if requested_qwen_clone_method not in ("openvoice", "native"):
            raise ValueError(
                f"unknown qwen_clone_method {qwen_clone_method!r}; choose 'openvoice' or 'native'"
            )
        if requested_qwen_clone_method == "native" and requested_backend == "melo":
            raise ValueError("qwen_clone_method='native' conflicts with backend='melo' (native mode is Qwen3-TTS-only)")

        qwen_language = config.QWEN_LANGUAGES.get(language.upper())
        # Requesting native mode is an unambiguous signal to use the Qwen
        # backend, same as requested_backend == "qwen" explicitly.
        if requested_backend == "qwen" or requested_qwen_clone_method == "native":
            if qwen_language is None:
                raise ValueError(
                    f"Qwen3-TTS has no voice for language {language!r}; "
                    f"supported: {sorted(config.QWEN_LANGUAGES)}"
                )
            tts_backend = "qwen"
        elif requested_backend == "melo":
            if language.upper() not in config.MELO_LANGUAGES:
                raise ValueError(
                    f"MeloTTS has no voice for language {language!r}; "
                    f"supported: {sorted(config.MELO_LANGUAGES)} (or use backend='qwen'/'auto' for "
                    f"{sorted(config.QWEN_ONLY_LANGUAGES)})"
                )
            tts_backend = "melo"
        else:
            tts_backend = "qwen" if language.upper() in config.QWEN_ONLY_LANGUAGES else "melo"

        use_native = tts_backend == "qwen" and requested_qwen_clone_method == "native"
        effective_qwen_clone_method = requested_qwen_clone_method if tts_backend == "qwen" else None

        # "auto" placeholder lets a cache hit be served without loading any
        # model at all; the concrete key actually used is recorded in the
        # saved metadata once resolved below. `speed`/`tau`/`base_speaker_key`
        # are normalized out wherever they can't actually affect the output
        # (Qwen has no speed control; native mode never touches OpenVoice's
        # conversion or any base-speaker preset at all) — otherwise requests
        # that would produce identical audio wouldn't share a cache entry.
        canonical_params = {
            "engine_version": _ENGINE_VERSION,
            "text": text,
            "language": language,
            "tts_backend": tts_backend,
            "qwen_clone_method": effective_qwen_clone_method,
            "speed": speed if tts_backend == "melo" else None,
            "tau": None if use_native else tau,
            "base_speaker_key": None if use_native else (base_speaker_key or "auto"),
            "mix": resolved_mix,
        }
        cache_key = storage.compute_cache_key(canonical_params)

        if not force:
            cached = storage.find_cached(folder_name, cache_key)
            if cached is not None:
                return SynthesisResult(item=cached, cached=True)

        with tempfile.TemporaryDirectory(prefix="voicelab-") as tmp_str:
            tmp = Path(tmp_str)
            out_wav = tmp / "converted.wav"

            if use_native:
                resolved_base_speaker_key = None
                self._generate_qwen_native_clone(text, qwen_language, resolved_mix, out_wav)
                total_weight = sum(entry["weight"] for entry in resolved_mix)
                mix_detail = [
                    {**entry, "normalized_weight": entry["weight"] / total_weight}
                    for entry in resolved_mix
                ]
            else:
                target_se, mix_detail = embeddings.communal_embedding(self.converter, mix_entries)
                base_wav = tmp / "base.wav"

                if tts_backend == "qwen":
                    resolved_base_speaker_key, source_se = self._generate_base_audio_qwen(
                        text, qwen_language, base_speaker_key, base_wav,
                    )
                else:
                    resolved_base_speaker_key, source_se = self._generate_base_audio_melo(
                        text, language, speed, base_speaker_key, base_wav,
                    )

                self.converter.convert(
                    audio_src_path=str(base_wav),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=str(out_wav),
                    tau=tau,
                    message=config.DEFAULT_WATERMARK_MESSAGE,
                )

            metadata = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "text": text,
                "language": language,
                "requested_backend": backend,
                "tts_backend": tts_backend,
                "qwen_clone_method": effective_qwen_clone_method,
                "speed": speed if tts_backend == "melo" else None,  # Qwen3-TTS has no speed control
                "tau": None if use_native else tau,
                "requested_base_speaker_key": base_speaker_key,
                "resolved_base_speaker_key": resolved_base_speaker_key,
                "mix": mix_detail,
                "engine_version": _ENGINE_VERSION,
                "device": self.device,
                "cache_key": cache_key,
            }
            item = storage.save_synthetic(folder_name, out_wav, metadata)

        return SynthesisResult(item=item, cached=False)

    def synthesize_chorus(
        self,
        text: str,
        mix: MixSpec,
        *,
        language: str = config.DEFAULT_LANGUAGE,
        speed: float = config.DEFAULT_SPEED,
        tau: float = config.DEFAULT_TAU,
        base_speaker_key: Optional[str] = None,
        backend: Optional[str] = None,
        qwen_clone_method: Optional[str] = None,
        force: bool = False,
    ) -> SynthesisResult:
        """Chorus mode: clone `text` individually and in full for *every*
        entry in `mix` — no embedding blending at all, each voice keeps
        its own complete identity — then mix the resulting clips together
        as a simultaneous ensemble (summed waveforms, not concatenated),
        so several voices say the same thing at the same time rather than
        one blended voice saying it once. `mix` weights become each
        voice's relative volume in the final mix instead of a blend
        ratio. Each individual voice's clone is produced (and cached) via
        the normal `synthesize()` path, so it's reusable/inspectable on
        its own too.
        """
        mix_entries = [m if isinstance(m, MixEntry) else MixEntry(**m) for m in mix]
        if len(mix_entries) < 2:
            raise ValueError("chorus mode needs at least 2 speakers/recordings to be a chorus")

        folder_name = storage.mix_folder_name([m.speaker for m in mix_entries]) + "-chorus"

        resolved_mix = []
        for m in mix_entries:
            files = sorted(m.files) if m.files else sorted(p.name for p in storage.list_recordings(m.speaker))
            if not files:
                raise ValueError(f"speaker {m.speaker!r} has no recordings in {storage.speaker_dir(m.speaker)}")
            checksums = {f: storage.checksum_file(storage.speaker_dir(m.speaker) / f) for f in files}
            resolved_mix.append({
                "speaker": m.speaker, "weight": m.weight, "files": files, "checksums": checksums,
            })

        canonical_params = {
            "engine_version": _ENGINE_VERSION,
            "mode": "chorus",
            "text": text,
            "language": language,
            "backend": backend or "auto",
            "qwen_clone_method": qwen_clone_method or "openvoice",
            "speed": speed,
            "tau": tau,
            "base_speaker_key": base_speaker_key or "auto",
            "mix": resolved_mix,
        }
        cache_key = storage.compute_cache_key(canonical_params)

        if not force:
            cached = storage.find_cached(folder_name, cache_key)
            if cached is not None:
                return SynthesisResult(item=cached, cached=True)

        voice_clips: list[tuple[Path, float]] = []
        individual_results = []
        for m in mix_entries:
            single = self.synthesize(
                text, [MixEntry(speaker=m.speaker, weight=1.0, files=m.files)],
                language=language, speed=speed, tau=tau,
                base_speaker_key=base_speaker_key, backend=backend,
                qwen_clone_method=qwen_clone_method, force=force,
            )
            voice_clips.append((single.item.audio_path, m.weight))
            individual_results.append({
                "speaker": m.speaker, "weight": m.weight,
                "folder": single.item.folder, "id": single.item.id,
            })

        with tempfile.TemporaryDirectory(prefix="voicelab-chorus-") as tmp_str:
            out_wav = Path(tmp_str) / "chorus.wav"
            chorus.mix_chorus(voice_clips, out_wav)

            metadata = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mode": "chorus",
                "text": text,
                "language": language,
                "mix": resolved_mix,
                "individual_clones": individual_results,
                "engine_version": _ENGINE_VERSION,
                "device": self.device,
                "cache_key": cache_key,
            }
            item = storage.save_synthetic(folder_name, out_wav, metadata)

        return SynthesisResult(item=item, cached=False)


_shared_engine: Optional[OpenVoiceEngine] = None
_shared_engine_lock = threading.Lock()


def get_shared_engine() -> OpenVoiceEngine:
    """One OpenVoiceEngine reused across requests, for long-running
    processes (the API server, the web app).

    Without this, every request would construct a fresh engine — and for
    the Qwen backend specifically, that means spawning a brand new
    subprocess with its own freshly-loaded multi-GB model *per request*,
    with nothing to ever terminate the previous one: an unbounded leak of
    GPU memory and orphaned processes. A single shared, lazily-built
    instance keeps every backend's loaded model/converter/subprocess
    alive and reused for the life of the server instead.
    """
    global _shared_engine
    if _shared_engine is None:
        with _shared_engine_lock:
            if _shared_engine is None:
                _shared_engine = OpenVoiceEngine()
                atexit.register(_shared_engine.close)
    return _shared_engine
