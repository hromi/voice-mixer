"""Gradio web UI for voicelab: record/upload voices, clone/mix speech, and
browse the library of human recordings and cached synthetic clones.
"""

from __future__ import annotations

import json
from typing import Hashable

import gradio as gr

from . import config, storage
from .embeddings import MixEntry

MIX_MODE = "Speaker mix (weighted)"
PIN_MODE = "Pinpoint recording(s)"
BACKEND_CHOICES = ["Auto", "MeloTTS", "Qwen3-TTS"]
_BACKEND_VALUES = {"Auto": None, "MeloTTS": "melo", "Qwen3-TTS": "qwen"}
QWEN_METHOD_CHOICES = ["OpenVoice tone-color transfer", "Qwen native x-vector clone"]
_QWEN_METHOD_VALUES = {
    "OpenVoice tone-color transfer": "openvoice",
    "Qwen native x-vector clone": "native",
}
DEFAULT_BACKEND_CHOICE = "Qwen3-TTS"
DEFAULT_QWEN_METHOD_CHOICE = "Qwen native x-vector clone"

WEIGHT_MIN, WEIGHT_MAX, WEIGHT_STEP = 0, 100, 1

# Sliders for "speaker mix" and "pinpoint" mode are a fixed pool of real,
# stable components (mostly hidden), not built dynamically via @gr.render.
# @gr.render's state-triggered rebuild turned out unreliable in practice
# for this rebalancing pattern (stale reads at click time); a fixed pool
# read/written directly via ordinary .release() -> gr.update() is the
# standard, well-tested Gradio mechanism and has none of that ambiguity.
# The cap just needs to comfortably exceed any realistic number of
# contributors/recordings picked at once.
MAX_SPEAKER_SLOTS = 20
MAX_PIN_SLOTS = 20


def _speaker_choices() -> list[str]:
    return storage.list_speakers()


def _equal_weights(names: list[Hashable]) -> dict:
    if not names:
        return {}
    share = WEIGHT_MAX / len(names)
    return {name: share for name in names}


def _rebalance_weights(names: list[Hashable], old_weights: dict, changed_name: Hashable, new_value: float) -> dict:
    """Moving one slider redistributes the remainder across the others,
    preserving their relative proportions to each other — the classic
    "budget allocation" pattern, so the sliders always sum back to
    WEIGHT_MAX after any single change."""
    new_value = max(WEIGHT_MIN, min(WEIGHT_MAX, new_value))
    result = {changed_name: new_value}
    others = [n for n in names if n != changed_name]
    if not others:
        return result

    remaining = WEIGHT_MAX - new_value
    others_old_sum = sum(old_weights.get(n, 0.0) for n in others)
    for n in others:
        share = (old_weights.get(n, 0.0) / others_old_sum) if others_old_sum > 1e-9 else (1.0 / len(others))
        result[n] = remaining * share
    return result


def _apply_speaker_slots(names: list[str]) -> list:
    """gr.update(...) for every speaker-slot slider: the first len(names)
    become visible, labeled, and equal-weighted; the rest hidden."""
    names = names[:MAX_SPEAKER_SLOTS]
    equal = _equal_weights(names)
    updates = []
    for i in range(MAX_SPEAKER_SLOTS):
        if i < len(names):
            updates.append(gr.update(visible=True, label=names[i], value=equal[names[i]]))
        else:
            updates.append(gr.update(visible=False, value=0))
    return updates


def _apply_pin_slots(pairs: list[tuple[str, str]]) -> list:
    pairs = pairs[:MAX_PIN_SLOTS]
    equal = _equal_weights(pairs)
    updates = []
    for i in range(MAX_PIN_SLOTS):
        if i < len(pairs):
            updates.append(gr.update(visible=True, label=_recording_label(*pairs[i]), value=equal[pairs[i]]))
        else:
            updates.append(gr.update(visible=False, value=0))
    return updates


def on_mix_mode_change(mode: str):
    is_pin = mode == PIN_MODE
    return gr.update(visible=not is_pin), gr.update(visible=is_pin)


def _effective_backend(language: str, backend: str) -> str:
    resolved = _BACKEND_VALUES.get(backend)
    if resolved:
        return resolved
    return "qwen" if language.upper() in config.QWEN_ONLY_LANGUAGES else "melo"


def on_qwen_context_change(language: str, backend: str, qwen_method: str):
    """Keeps three controls in sync with the current language/backend/method
    combo: the "Qwen clone method" selector only makes sense once Qwen is
    actually the effective backend; "native" mode bypasses OpenVoice
    entirely, so the base-speaker preset (nothing to pick — no preset voice
    is used) and the conversion-strength slider (nothing to convert) are
    both meaningless there and hidden rather than left sitting inert.
    """
    is_qwen = _effective_backend(language, backend) == "qwen"
    is_native = is_qwen and _QWEN_METHOD_VALUES.get(qwen_method) == "native"

    method_update = gr.update(visible=is_qwen)

    if is_native:
        base_speaker_update = gr.update(choices=[""], value="", visible=False)
    elif is_qwen:
        base_speaker_update = gr.update(choices=[""] + config.QWEN_SPEAKERS, value="", visible=True)
    else:
        choices = config.MELO_BASE_SPEAKERS.get(language.upper(), [])
        base_speaker_update = gr.update(choices=[""] + choices, value="", visible=True)

    tau_update = gr.update(visible=not is_native)

    return method_update, base_speaker_update, tau_update


def _recording_rows(speaker: str) -> list[list[str]]:
    return [
        [p.name, storage.recording_metadata(speaker, p.name).get("note", "")]
        for p in storage.list_recordings(speaker)
    ]


def _recording_label(speaker: str, filename: str) -> str:
    note = storage.recording_metadata(speaker, filename).get("note", "").strip()
    return f"[{speaker}] {note}" if note else f"[{speaker}] {filename}"


def _all_recording_choices() -> list[tuple[str, str]]:
    """(label, value) pairs across every speaker's every recording, value
    being a "speaker/filename" composite key (both are already sanitized
    to a safe charset, so "/" can't collide with either part)."""
    choices = []
    for speaker in storage.list_speakers():
        for p in storage.list_recordings(speaker):
            choices.append((_recording_label(speaker, p.name), f"{speaker}/{p.name}"))
    return choices


def save_recording(speaker: str, audio_path: str, note: str):
    if not speaker or not speaker.strip():
        return ("Please enter a speaker name.", gr.update(), gr.update(), gr.update(),
                *[gr.update()] * MAX_SPEAKER_SLOTS)
    if not audio_path:
        return ("Record or upload audio first.", gr.update(), gr.update(), gr.update(),
                *[gr.update()] * MAX_SPEAKER_SLOTS)
    speaker = speaker.strip()
    dest = storage.add_recording(speaker, audio_path, source="web", note=note.strip() if note else None)
    names = _speaker_choices()
    return (
        f"Saved {dest.name} for {speaker}.",
        gr.update(choices=names, value=speaker),
        _recording_rows(speaker),
        names,
        *_apply_speaker_slots(names),
    )


def run_clone(text: str, mix_mode: str, speaker_names: list[str], pin_pairs: list[tuple[str, str]], *rest):
    from .engine import get_shared_engine  # deferred: heavy ML import

    speaker_values = rest[:MAX_SPEAKER_SLOTS]
    pin_values = rest[MAX_SPEAKER_SLOTS:MAX_SPEAKER_SLOTS + MAX_PIN_SLOTS]
    language, backend, qwen_method, speed, tau, base_speaker, force = rest[MAX_SPEAKER_SLOTS + MAX_PIN_SLOTS:]

    if not text or not text.strip():
        raise gr.Error("Enter some text to speak.")

    if mix_mode == PIN_MODE:
        active_pairs = (pin_pairs or [])[:MAX_PIN_SLOTS]
        mix = [
            MixEntry(speaker=spk, weight=weight, files=[filename])
            for (spk, filename), weight in zip(active_pairs, pin_values)
            if weight > 0
        ]
        if not mix:
            raise gr.Error("Pick at least one recording and give it a nonzero weight.")
    else:
        active_names = (speaker_names or [])[:MAX_SPEAKER_SLOTS]
        mix = [MixEntry(speaker=name, weight=weight) for name, weight in zip(active_names, speaker_values) if weight > 0]
        if not mix:
            raise gr.Error("No speakers with a nonzero weight — record a voice first, or move a slider up.")

    engine = get_shared_engine()
    result = engine.synthesize(
        text.strip(), mix, language=language, speed=speed, tau=tau,
        base_speaker_key=(base_speaker or "").strip() or None, backend=_BACKEND_VALUES.get(backend),
        qwen_clone_method=_QWEN_METHOD_VALUES.get(qwen_method), force=force,
    )
    status = "Served from cache." if result.cached else "Generated."
    return str(result.item.audio_path), f"{status} ({result.item.folder}/{result.item.id})", json.dumps(result.item.metadata, indent=2)


def refresh_speaker_recordings(speaker: str):
    if not speaker:
        return []
    return _recording_rows(speaker)


def play_selected_recording(speaker: str, evt: gr.SelectData):
    if not speaker:
        return None, None
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    recordings = storage.list_recordings(speaker)
    if row_idx is None or row_idx >= len(recordings):
        return None, None
    selected = recordings[row_idx]
    return str(selected), selected.name


def delete_selected_recording(speaker: str, filename: str):
    if not speaker or not filename:
        return "Nothing selected to delete.", gr.update(), gr.update(), None, None
    storage.remove_recording(speaker, filename)
    return (
        f"Deleted {filename}.",
        gr.update(choices=_speaker_choices()),
        _recording_rows(speaker),
        None,
        None,
    )


def _synthetic_item_label(item: storage.SyntheticItem) -> str:
    language = item.metadata.get("language", "?")
    text = (item.metadata.get("text") or "").strip().replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."
    return f"[{language}] {text}  ({item.id})"


def refresh_synthetic_items(folder: str):
    if not folder:
        return gr.update(choices=[]), None, "{}"
    items = storage.list_synthetic_items(folder)
    choices = [(_synthetic_item_label(item), item.id) for item in items]
    return gr.update(choices=choices), None, "{}"


def load_synthetic_item(folder: str, item_id: str):
    if not folder or not item_id:
        return None, "{}"
    item = storage.get_synthetic_item(folder, item_id)
    if item is None:
        return None, "{}"
    return str(item.audio_path), json.dumps(item.metadata, indent=2)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="voicelab") as demo:
        gr.Markdown(
            "# voicelab\n"
            "Communal voice cloning on top of OpenVoice. Record or upload reference "
            "voices, then clone or **cross over multiple speakers** into new speech."
        )

        # Page-level state shared between the Record and Clone tabs, so a
        # recording added in one place is immediately available as a
        # slider in the other without needing a page reload. These only
        # track *identity* (which speaker/recording occupies which slot) —
        # the weights themselves live directly on the slider components,
        # read straight from them wherever needed (no separate State to
        # keep in sync).
        speaker_names_state = gr.State([])
        pin_selected_state = gr.State([])

        with gr.Tab("Record / Upload Voice"):
            with gr.Row():
                with gr.Column():
                    speaker_name = gr.Textbox(label="Speaker name", placeholder="e.g. SolarPunk0")
                    rec_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Recording")
                    note_in = gr.Textbox(
                        label="Note / transcript (optional)",
                        placeholder="What was said, or any other note about this recording",
                    )
                    save_btn = gr.Button("Save recording", variant="primary")
                    save_status = gr.Markdown()
                with gr.Column():
                    existing_speaker = gr.Dropdown(choices=[], label="Existing speaker recordings")
                    existing_files = gr.Dataframe(headers=["file", "note"], datatype=["str", "str"], interactive=False)
                    gr.Markdown("Click a row to play it.")
                    selected_audio = gr.Audio(label="Selected recording", type="filepath", interactive=False)
                    selected_filename = gr.State(None)
                    delete_btn = gr.Button("Delete selected recording", variant="stop")
                    with gr.Row(visible=False) as delete_confirm_row:
                        confirm_delete_btn = gr.Button("Yes, delete it", variant="stop")
                        cancel_delete_btn = gr.Button("Cancel")
                    delete_status = gr.Markdown()

            existing_speaker.change(refresh_speaker_recordings, inputs=existing_speaker, outputs=existing_files)
            existing_files.select(
                play_selected_recording, inputs=existing_speaker,
                outputs=[selected_audio, selected_filename],
            )

            def ask_delete_confirmation(filename):
                if not filename:
                    return "Click a recording in the table first.", gr.update(visible=False)
                return f"Delete **{filename}**? This cannot be undone.", gr.update(visible=True)

            delete_btn.click(
                ask_delete_confirmation, inputs=selected_filename,
                outputs=[delete_status, delete_confirm_row],
            )
            cancel_delete_btn.click(
                lambda: ("", gr.update(visible=False)),
                outputs=[delete_status, delete_confirm_row],
            )

        with gr.Tab("Clone / Communal Mix"):
            gr.Markdown(
                "Two ways to pick the source voice: a **speaker mix** — every known "
                "speaker gets a weight slider, move one and the others rebalance to "
                "compensate — for a communal crossover; or **pinpointing** exact "
                "recordings (from any speaker), each with its own weight slider, "
                "bypassing per-speaker averaging entirely."
            )
            with gr.Row():
                with gr.Column():
                    text_in = gr.Textbox(label="Text to speak", lines=4)
                    mix_mode_in = gr.Radio([MIX_MODE, PIN_MODE], value=MIX_MODE, label="Source voice mode")

                    with gr.Group(visible=True) as mix_group:
                        gr.Markdown("Drag a speaker's weight — the others rebalance automatically.")
                        speaker_sliders = [
                            gr.Slider(WEIGHT_MIN, WEIGHT_MAX, value=0, step=WEIGHT_STEP, visible=False,
                                      label=f"speaker-slot-{i}")
                            for i in range(MAX_SPEAKER_SLOTS)
                        ]

                    with gr.Group(visible=False) as pin_group:
                        pin_multiselect_in = gr.Dropdown(
                            choices=[], multiselect=True, label="Recordings to pinpoint",
                            info="Pick any recordings from any speaker(s); each gets its own weight below",
                        )
                        pin_sliders = [
                            gr.Slider(WEIGHT_MIN, WEIGHT_MAX, value=0, step=WEIGHT_STEP, visible=False,
                                      label=f"pin-slot-{i}")
                            for i in range(MAX_PIN_SLOTS)
                        ]

                    with gr.Row():
                        language_in = gr.Dropdown(
                            ["EN", "EN_NEWEST", "ES", "FR", "ZH", "JP", "KR",
                             "DE", "RU", "PT", "IT"],
                            value=config.DEFAULT_LANGUAGE, label="Language",
                            info="DE/RU/PT/IT use Qwen3-TTS (MeloTTS has no voice for these)",
                        )
                        backend_in = gr.Radio(
                            BACKEND_CHOICES, value=DEFAULT_BACKEND_CHOICE, label="TTS backend",
                            info="Auto picks Qwen3-TTS only where MeloTTS has no voice; "
                                 "EN/ES/FR/ZH/JP/KR support either, force one to compare",
                        )
                    qwen_method_in = gr.Radio(
                        QWEN_METHOD_CHOICES, value=DEFAULT_QWEN_METHOD_CHOICE, label="Qwen clone method",
                        visible=True,
                        info="OpenVoice: Qwen speaks a preset voice, then OpenVoice retargets it to your "
                             "mix (works for any backend). Native: Qwen3-TTS's own x-vector cloning "
                             "generates directly in the blended voice — no OpenVoice step at all.",
                    )
                    with gr.Row():
                        base_speaker_in = gr.Dropdown(
                            choices=[""] + config.QWEN_SPEAKERS, value="", visible=False,
                            label="Base speaker",
                            info="The stock voice that gets converted to your target speaker(s); "
                                 "blank picks one automatically. Options depend on Language/backend above.",
                        )
                    with gr.Row():
                        speed_in = gr.Slider(0.5, 2.0, value=config.DEFAULT_SPEED, step=0.05, label="Speed")
                        tau_in = gr.Slider(0.0, 1.0, value=config.DEFAULT_TAU, step=0.05, visible=False,
                                            label="Conversion strength (tau)")
                    force_in = gr.Checkbox(value=False, label="Force regenerate (ignore cache)")
                    clone_btn = gr.Button("Clone voice", variant="primary")
                with gr.Column():
                    clone_audio = gr.Audio(label="Result", type="filepath")
                    clone_status = gr.Markdown()
                    clone_meta = gr.Code(label="Metadata", language="json")

            mix_mode_in.change(on_mix_mode_change, inputs=mix_mode_in, outputs=[mix_group, pin_group])

            # Speaker mix: one .release() handler per slot. Reads current
            # values straight off the live slider components (both the
            # just-dragged slot's new value and every other slot's
            # unchanged value) — no separate state to fall out of sync
            # with what's actually displayed.
            def _make_speaker_slot_handler(idx: int):
                def _handler(names, *values):
                    n = len(names)
                    if idx >= n:
                        return [gr.update()] * MAX_SPEAKER_SLOTS
                    current = dict(zip(names, values[:n]))
                    changed_name = names[idx]
                    new_weights = _rebalance_weights(names, current, changed_name, current[changed_name])
                    return [
                        gr.update(value=new_weights[names[i]]) if i < n else gr.update()
                        for i in range(MAX_SPEAKER_SLOTS)
                    ]
                return _handler

            for idx, slider in enumerate(speaker_sliders):
                slider.release(
                    _make_speaker_slot_handler(idx), inputs=[speaker_names_state, *speaker_sliders],
                    outputs=speaker_sliders,
                )

            def on_pin_selection_change(selected_keys: list[str]):
                pairs = [tuple(key.split("/", 1)) for key in (selected_keys or [])]
                return (pairs, *_apply_pin_slots(pairs))

            pin_multiselect_in.change(
                on_pin_selection_change, inputs=pin_multiselect_in,
                outputs=[pin_selected_state, *pin_sliders],
            )

            def _make_pin_slot_handler(idx: int):
                def _handler(pairs, *values):
                    n = len(pairs)
                    if idx >= n:
                        return [gr.update()] * MAX_PIN_SLOTS
                    pairs = [tuple(p) for p in pairs]
                    current = dict(zip(pairs, values[:n]))
                    changed_pair = pairs[idx]
                    new_weights = _rebalance_weights(pairs, current, changed_pair, current[changed_pair])
                    return [
                        gr.update(value=new_weights[pairs[i]]) if i < n else gr.update()
                        for i in range(MAX_PIN_SLOTS)
                    ]
                return _handler

            for idx, slider in enumerate(pin_sliders):
                slider.release(
                    _make_pin_slot_handler(idx), inputs=[pin_selected_state, *pin_sliders],
                    outputs=pin_sliders,
                )

            save_btn.click(
                save_recording, inputs=[speaker_name, rec_audio, note_in],
                outputs=[save_status, existing_speaker, existing_files, speaker_names_state, *speaker_sliders],
            )
            confirm_delete_btn.click(
                delete_selected_recording, inputs=[existing_speaker, selected_filename],
                outputs=[delete_status, existing_speaker, existing_files, selected_audio, selected_filename],
            ).then(
                lambda: gr.update(visible=False), outputs=delete_confirm_row,
            ).then(
                lambda: (_speaker_choices(), *_apply_speaker_slots(_speaker_choices())),
                outputs=[speaker_names_state, *speaker_sliders],
            )

            _qwen_context_inputs = [language_in, backend_in, qwen_method_in]
            _qwen_context_outputs = [qwen_method_in, base_speaker_in, tau_in]
            language_in.change(on_qwen_context_change, inputs=_qwen_context_inputs, outputs=_qwen_context_outputs)
            backend_in.change(on_qwen_context_change, inputs=_qwen_context_inputs, outputs=_qwen_context_outputs)
            qwen_method_in.change(on_qwen_context_change, inputs=_qwen_context_inputs, outputs=_qwen_context_outputs)

            clone_btn.click(
                run_clone,
                inputs=[text_in, mix_mode_in, speaker_names_state, pin_selected_state,
                        *speaker_sliders, *pin_sliders,
                        language_in, backend_in, qwen_method_in, speed_in, tau_in, base_speaker_in, force_in],
                outputs=[clone_audio, clone_status, clone_meta],
            )

        with gr.Tab("Library"):
            gr.Markdown("Browse cached synthetic clones by speaker/mix folder.")
            with gr.Row():
                with gr.Column():
                    folder_in = gr.Dropdown(choices=[], label="Speaker / mix folder")
                    refresh_btn = gr.Button("Refresh folders")
                    item_in = gr.Dropdown(choices=[], label="Clone id")
                with gr.Column():
                    lib_audio = gr.Audio(label="Audio", type="filepath")
                    lib_meta = gr.Code(label="Metadata", language="json")

            refresh_btn.click(lambda: gr.update(choices=storage.list_synthetic_folders()), outputs=folder_in)
            folder_in.change(refresh_synthetic_items, inputs=folder_in, outputs=[item_in, lib_audio, lib_meta])
            item_in.change(load_synthetic_item, inputs=[folder_in, item_in], outputs=[lib_audio, lib_meta])

        # Dropdown/slider choices above are otherwise frozen at the moment
        # this Blocks graph was built (i.e. process startup) — anyone who
        # records/clones after that and then just reloads the page would
        # see stale/empty controls despite the files existing on disk.
        # Recompute them fresh on every page load instead.
        def _on_load():
            names = _speaker_choices()
            return (
                gr.update(choices=names),
                gr.update(choices=storage.list_synthetic_folders()),
                names,
                gr.update(choices=_all_recording_choices()),
                *_apply_speaker_slots(names),
            )

        demo.load(
            _on_load,
            outputs=[existing_speaker, folder_in, speaker_names_state, pin_multiselect_in, *speaker_sliders],
        )
        demo.load(on_qwen_context_change, inputs=_qwen_context_inputs, outputs=_qwen_context_outputs)

        gr.Markdown(
            "---\n"
            "[voice-mixer on GitHub](https://github.com/hromi/voice-mixer) · "
            "part of the [udk.ai](https://udk.ai) suite",
            elem_id="footer",
        )

    return demo


def build_asgi_app(mount_path: str = "/tts") -> "FastAPI":
    """FastAPI app with the Gradio UI mounted at both "/" and `mount_path`.

    Mounted twice because we're behind a reverse proxy (amsel.udk.ai/tts ->
    this process) and it isn't known in advance whether the proxy strips
    the "/tts" prefix before forwarding or forwards it verbatim — mounting
    at both makes the app reachable either way. Each mount gets its own
    Blocks instance since Gradio's mount call is not safe to repeat on the
    same instance.
    """
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    app = FastAPI(title="voicelab-web")
    if mount_path and mount_path != "/":
        @app.get(mount_path, include_in_schema=False)
        def _redirect_to_trailing_slash():
            return RedirectResponse(url=mount_path.rstrip("/") + "/")
    # The "/" mount has an empty path prefix, so it matches every incoming
    # path (Starlette routes are matched in registration order) — it must
    # be registered *after* the more specific mount_path, or it would
    # swallow every request before mount_path is ever reached.
    auth = (config.WEB_AUTH_USERNAME, config.get_web_auth_password())
    if mount_path and mount_path != "/":
        gr.mount_gradio_app(app, build_app(), path=mount_path, auth=auth)
    gr.mount_gradio_app(app, build_app(), path="/", auth=auth)
    return app


if __name__ == "__main__":
    build_app().launch(auth=(config.WEB_AUTH_USERNAME, config.get_web_auth_password()))
