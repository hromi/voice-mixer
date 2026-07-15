"""Standalone Qwen3-TTS worker, run inside its own Python 3.10+ venv
(.venv-qwen/), since the qwen-tts package uses `X | None` union-type
syntax at module level that Python 3.9 (this project's main interpreter)
cannot parse.

Talks newline-delimited JSON over stdin/stdout so voicelab/engine.py can
keep one persistent subprocess alive across requests, instead of paying
the multi-second model-load cost on every synthesis call. Commands:

    {"cmd": "supported_speakers"}
    {"cmd": "supported_languages"}
    {"cmd": "generate_preset", "text", "language", "speaker", "out_path"}
        -> speaks `text` in one of the 9 built-in preset voices (loads the
           -CustomVoice checkpoint). This is the base utterance for the
           "openvoice" qwen_clone_method — OpenVoice converts it afterward.
    {"cmd": "generate_native_clone", "text", "language", "out_path",
     "mix": [{"files": [path, ...], "weight": w}, ...]}
        -> loads the -Base checkpoint (separate, lazy) and extracts a
           speaker x-vector from each file via
           model.extract_speaker_embedding (through create_voice_clone_prompt),
           averages per mix entry, blends entries by weight, and generates
           `text` directly conditioned on that blended embedding — no
           OpenVoice conversion step at all. Mirrors the exact same
           per-speaker-average / weighted-blend-across-speakers semantics
           as voicelab/embeddings.py, just computed with Qwen's own
           x-vector space instead of OpenVoice's tone-color embeddings.

Every response is {"ok": true, ...} or {"ok": false, "error": "..."}.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    # qwen-tts (and libraries it pulls in) print incidental status messages
    # straight to stdout (e.g. a "flash-attn is not installed" banner at
    # import time) — that would corrupt our line-based JSON protocol. Steal
    # the real stdout for protocol use only, and let everything else
    # (including our own accidental prints, third-party libs, stray
    # library logging) fall through to stderr instead.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    def send(obj: dict) -> None:
        protocol_out.write(json.dumps(obj) + "\n")
        protocol_out.flush()

    import torch
    from qwen_tts import Qwen3TTSModel

    custom_voice_model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    base_model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device = sys.argv[3] if len(sys.argv) > 3 else ("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Qwen3TTSModel.from_pretrained(custom_voice_model_id, device_map=device, dtype=torch.bfloat16)
    base_model_holder: dict = {}  # lazy: only load the ~4GB Base checkpoint if native mode is actually used

    def get_base_model():
        if "model" not in base_model_holder:
            base_model_holder["model"] = Qwen3TTSModel.from_pretrained(
                base_model_id, device_map=device, dtype=torch.bfloat16,
            )
        return base_model_holder["model"]

    def speaker_embedding(base_model, files: list[str]):
        embeddings = []
        for f in files:
            items = base_model.create_voice_clone_prompt(ref_audio=f, x_vector_only_mode=True)
            embeddings.append(items[0].ref_spk_embedding)
        total = embeddings[0]
        for e in embeddings[1:]:
            total = total + e
        return total / len(embeddings)

    def blend_embeddings(weighted: list[tuple]):
        total_weight = sum(w for _, w in weighted)
        result = None
        for embedding, weight in weighted:
            term = embedding * (weight / total_weight)
            result = term if result is None else result + term
        return result

    send({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "supported_speakers":
                result = {"ok": True, "result": model.get_supported_speakers() or []}
            elif cmd == "supported_languages":
                result = {"ok": True, "result": model.get_supported_languages() or []}
            elif cmd == "generate_preset":
                import soundfile as sf

                wavs, sr = model.generate_custom_voice(
                    text=req["text"], language=req["language"], speaker=req["speaker"],
                )
                sf.write(req["out_path"], wavs[0], sr)
                result = {"ok": True}
            elif cmd == "generate_native_clone":
                import soundfile as sf

                base_model = get_base_model()
                weighted = [
                    (speaker_embedding(base_model, entry["files"]), entry["weight"])
                    for entry in req["mix"]
                ]
                blended = blend_embeddings(weighted)
                voice_clone_prompt = {
                    "ref_code": [None],
                    "ref_spk_embedding": [blended],
                    "x_vector_only_mode": [True],
                    "icl_mode": [False],
                }
                wavs, sr = base_model.generate_voice_clone(
                    text=req["text"], language=req["language"], voice_clone_prompt=voice_clone_prompt,
                )
                sf.write(req["out_path"], wavs[0], sr)
                result = {"ok": True}
            else:
                result = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as exc:  # noqa: BLE001 - report every failure back to the parent process
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        send(result)


if __name__ == "__main__":
    main()
