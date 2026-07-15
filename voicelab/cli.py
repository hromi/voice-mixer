"""Command-line interface for voicelab.

    voicelab voices list
    voicelab voices add SolarPunk0 recording1.wav recording2.wav --note "what was said"
    voicelab clone "Hello there" --mix SolarPunk0
    voicelab clone "Hello there" --mix "SolarPunk0:0.6,JaneDoe:0.4"
    voicelab clone "Hello there" --mix "SolarPunk0[recording1.wav|recording2.wav]"
    voicelab synthetic list
    voicelab serve api
    voicelab serve web
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import config, storage
from .mixspec import parse_mix_spec

app = typer.Typer(help="Communal voice cloning on top of OpenVoice.")
voices_app = typer.Typer(help="Manage human reference recordings.")
synthetic_app = typer.Typer(help="Browse cached synthetic (cloned) audio.")
serve_app = typer.Typer(help="Run the HTTP API or the Gradio web app.")
app.add_typer(voices_app, name="voices")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(serve_app, name="serve")


@voices_app.command("list")
def voices_list():
    """List known speakers and how many recordings each has."""
    speakers = storage.list_speakers()
    if not speakers:
        typer.echo(f"No speakers yet. Add one with: voicelab voices add <name> <file...>")
        return
    for name in speakers:
        n = len(storage.list_recordings(name))
        typer.echo(f"{name}\t{n} recording(s)")


@voices_app.command("add")
def voices_add(
    speaker: str = typer.Argument(..., help="Speaker name, e.g. SolarPunk0"),
    files: list[Path] = typer.Argument(..., help="One or more audio files to add"),
    note: Optional[str] = typer.Option(None, "--note", "-n",
                                        help="Free-text note/transcript, applied to all files added"),
):
    """Add one or more recordings for a speaker."""
    for f in files:
        if not f.exists():
            typer.echo(f"skip (not found): {f}", err=True)
            continue
        dest = storage.add_recording(speaker, f, original_filename=f.name, source="upload", note=note)
        typer.echo(f"added {dest}")


@voices_app.command("remove")
def voices_remove(speaker: str, filename: str):
    """Remove a single recording from a speaker's folder."""
    storage.remove_recording(speaker, filename)
    typer.echo(f"removed {filename} from {speaker}")


@app.command()
def clone(
    text: str = typer.Argument(..., help="Text to speak in the cloned voice"),
    mix: str = typer.Option(..., "--mix", "-m",
                             help='Speaker mix, e.g. "SolarPunk0", "SolarPunk0:0.6,JaneDoe:0.4", or '
                                  '"SolarPunk0[file1.wav|file2.wav]:0.6" to pin exact source recordings'),
    language: str = typer.Option(config.DEFAULT_LANGUAGE, "--language", "-l",
                                  help="EN/ES/FR/ZH/JP/KR use MeloTTS; DE/RU/PT/IT use Qwen3-TTS "
                                       "(MeloTTS has no voice for those)"),
    speed: float = typer.Option(config.DEFAULT_SPEED, "--speed", help="MeloTTS only; no effect for DE/RU/PT/IT"),
    tau: float = typer.Option(config.DEFAULT_TAU, "--tau", help="Voice conversion strength"),
    base_speaker: Optional[str] = typer.Option(None, "--base-speaker",
                                                help="Base TTS speaker key: a MeloTTS speaker id (e.g. EN-Default) "
                                                     "or a Qwen3-TTS preset name (e.g. Ryan) for DE/RU/PT/IT "
                                                     "(auto if omitted)"),
    backend: Optional[str] = typer.Option(None, "--backend",
                                           help="Force the base-TTS engine: 'melo' or 'qwen' (auto by language "
                                                "if omitted). EN/ES/FR/ZH/JP/KR are covered by both."),
    qwen_clone_method: Optional[str] = typer.Option(
        None, "--qwen-clone-method",
        help="Only matters for the Qwen backend: 'openvoice' (default, preset voice + OpenVoice "
             "tone-color conversion) or 'native' (Qwen3-TTS's own x-vector cloning, no OpenVoice "
             "step at all — implies backend='qwen')",
    ),
    force: bool = typer.Option(False, "--force", help="Recompute even if a cached result exists"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Also copy the result to this path"),
):
    """Synthesize (or fetch from cache) a communal voice clone."""
    from .engine import OpenVoiceEngine  # deferred: heavy ML import

    mix_entries = parse_mix_spec(mix)
    engine = OpenVoiceEngine()
    result = engine.synthesize(
        text, mix_entries, language=language, speed=speed, tau=tau,
        base_speaker_key=base_speaker, backend=backend, qwen_clone_method=qwen_clone_method, force=force,
    )
    typer.echo(f"{'cached' if result.cached else 'generated'}: {result.item.audio_path}")
    typer.echo(json.dumps(result.item.metadata, indent=2))
    if out:
        out.write_bytes(result.item.audio_path.read_bytes())
        typer.echo(f"copied to {out}")


@synthetic_app.command("list")
def synthetic_list(folder: Optional[str] = typer.Argument(None, help="Speaker or Speaker+Speaker2 folder")):
    """List cached synthetic clones, optionally scoped to one speaker/mix folder."""
    folders = [folder] if folder else storage.list_synthetic_folders()
    if not folders:
        typer.echo("No synthetic audio yet.")
        return
    for f in folders:
        for item in storage.list_synthetic_items(f):
            typer.echo(f"{f}/{item.id}\t{item.metadata.get('text', '')!r}")


@synthetic_app.command("show")
def synthetic_show(folder: str, item_id: str):
    """Print full metadata for one cached clone."""
    item = storage.get_synthetic_item(folder, item_id)
    if item is None:
        typer.echo("not found", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(item.metadata, indent=2))


@serve_app.command("api")
def serve_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run("voicelab.api:app", host=host, port=port, reload=reload)


@serve_app.command("web")
def serve_web(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    mount_path: str = typer.Option("/tts", help='Extra path to mount the UI at, for reverse proxies (e.g. "/tts")'),
):
    """Run the Gradio web app (mounted at both "/" and --mount-path)."""
    if share:
        from .webapp import build_app
        build_app().launch(
            server_name=host, server_port=port, share=True,
            auth=(config.WEB_AUTH_USERNAME, config.get_web_auth_password()),
        )
        return

    import uvicorn
    from .webapp import build_asgi_app
    uvicorn.run(build_asgi_app(mount_path), host=host, port=port)


if __name__ == "__main__":
    app()
