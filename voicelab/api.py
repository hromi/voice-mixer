"""HTTP API for voicelab.

Endpoints
---------
GET    /speakers                                  list speakers + recording counts
POST   /speakers/{speaker}/recordings              upload a recording (multipart file)
GET    /speakers/{speaker}/recordings              list a speaker's recordings
DELETE /speakers/{speaker}/recordings/{filename}   remove a recording
POST   /synthesize                                 clone/mix text into speech (with caching)
GET    /synthetic                                  list speaker/mix folders
GET    /synthetic/{folder}                         list cached clones in a folder
GET    /synthetic/{folder}/{item_id}                metadata for one cached clone
GET    /synthetic/{folder}/{item_id}/audio          the clone's audio (audio/wav)
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config, storage
from .engine import CheckpointsNotFound
from .storage import InvalidSpeakerName

app = FastAPI(title="voicelab", description="Communal voice cloning on top of OpenVoice.")


class MixItem(BaseModel):
    speaker: str
    weight: float = 1.0
    files: Optional[list[str]] = None


class SynthesizeRequest(BaseModel):
    text: str
    mix: list[MixItem] = Field(..., min_length=1)
    language: str = config.DEFAULT_LANGUAGE
    speed: float = config.DEFAULT_SPEED
    tau: float = config.DEFAULT_TAU
    base_speaker_key: Optional[str] = None
    backend: Optional[str] = None  # "melo" | "qwen" | None (auto by language)
    qwen_clone_method: Optional[str] = None  # "openvoice" | "native" | None (openvoice); implies backend="qwen"
    chorus: bool = False  # clone each mix entry individually and mix as a simultaneous ensemble, not a blend
    force: bool = False


class SynthesizeResponse(BaseModel):
    folder: str
    id: str
    cached: bool
    audio_url: str
    metadata: dict


def _handle_common_errors(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except InvalidSpeakerName as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CheckpointsNotFound as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/speakers")
def list_speakers():
    return [
        {"name": name, "recordings": len(storage.list_recordings(name))}
        for name in storage.list_speakers()
    ]


@app.get("/speakers/{speaker}/recordings")
def list_recordings(speaker: str):
    return _handle_common_errors(
        lambda: [
            {"filename": p.name, "note": storage.recording_metadata(speaker, p.name).get("note", "")}
            for p in storage.list_recordings(speaker)
        ]
    )


@app.post("/speakers/{speaker}/recordings", status_code=201)
async def upload_recording(speaker: str, file: UploadFile = File(...), note: Optional[str] = Form(None)):
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        dest = _handle_common_errors(
            storage.add_recording, speaker, tmp_path,
            original_filename=file.filename, source="upload", move=True, note=note,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"speaker": speaker, "filename": dest.name}


@app.delete("/speakers/{speaker}/recordings/{filename}", status_code=204)
def delete_recording(speaker: str, filename: str):
    _handle_common_errors(storage.remove_recording, speaker, filename)


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    from .engine import get_shared_engine  # deferred: heavy ML import

    engine = get_shared_engine()
    mix = [m.model_dump() for m in req.mix]
    synth_fn = engine.synthesize_chorus if req.chorus else engine.synthesize
    result = _handle_common_errors(
        synth_fn, req.text, mix, language=req.language, speed=req.speed,
        tau=req.tau, base_speaker_key=req.base_speaker_key, backend=req.backend,
        qwen_clone_method=req.qwen_clone_method, force=req.force,
    )
    return SynthesizeResponse(
        folder=result.item.folder,
        id=result.item.id,
        cached=result.cached,
        audio_url=f"/synthetic/{result.item.folder}/{result.item.id}/audio",
        metadata=result.item.metadata,
    )


@app.get("/synthetic")
def list_synthetic_folders():
    return storage.list_synthetic_folders()


@app.get("/synthetic/{folder}")
def list_synthetic_items(folder: str):
    return [
        {"id": item.id, "text": item.metadata.get("text"), "created_at": item.metadata.get("created_at")}
        for item in _handle_common_errors(storage.list_synthetic_items, folder)
    ]


@app.get("/synthetic/{folder}/{item_id}")
def get_synthetic_metadata(folder: str, item_id: str):
    item = _handle_common_errors(storage.get_synthetic_item, folder, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="not found")
    return item.metadata


@app.get("/synthetic/{folder}/{item_id}/audio")
def get_synthetic_audio(folder: str, item_id: str):
    item = _handle_common_errors(storage.get_synthetic_item, folder, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(item.audio_path, media_type="audio/wav")
