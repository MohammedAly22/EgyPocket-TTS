"""Export a trained run into a self-contained CPU inference bundle.

    export/<name>/
        config.yaml               pocket-tts config (paths relative to this folder)
        model.safetensors         flow_lm (EMA weights) + Mimi codec
        tokenizer.json            character tokenizer
        ta_marbuta_lexicon.json   ه -> ة rewrites learned from the corpus
        voice.wav / voice.safetensors   default voice prompt (a held-out clip prefix)
        info.json

The folder can be copied anywhere (laptop, server) and loaded with
`EgyPocketTTS(<folder>)`; nothing else from the training setup is needed.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import soundfile
import torch
import yaml

from egypocket.data.align import _load_audio, _resample
from egypocket.data.jsonl import read_json, read_jsonl
from egypocket.modelcfg import bundle_config
from egypocket.paths import DataPaths


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir}")
    return ckpts[-1]


def export_weights(checkpoint: Path, model_config: Path, out_path: Path, use_ema: bool = True):
    """flow_lm (EMA overlaid) from a training checkpoint + Mimi from the config's bundle."""
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = dict(payload["model"])
    if use_ema and payload.get("ema"):
        state.update(payload["ema"])
    flow = {k: v.float().contiguous() for k, v in state.items() if k.startswith("flow_lm.")}
    if not flow:
        raise ValueError(f"no flow_lm weights in {checkpoint}")
    config = load_config(model_config)
    base = safetensors.torch.load_file(str(download_if_necessary(str(config.weights_path))))
    mimi = {k: v.contiguous() for k, v in base.items() if k.startswith("mimi.")}
    safetensors.torch.save_file({**flow, **mimi}, str(out_path))
    return int(payload.get("step", 0))


def _default_voice_audio(paths: DataPaths, sample_rate: int) -> np.ndarray:
    """Prefix of the first validation utterance, cut at a word boundary (a pause)."""
    cache = paths.cache_dir("valid")
    index = read_jsonl(cache / "index.jsonl")
    meta = read_json(cache / "meta.json")
    aligned = {r["id"]: r for r in read_jsonl(paths.aligned("valid"))}
    utt = index[0]
    cut_frame = utt["cuts"][-1][0]
    cut_sec = cut_frame / float(meta["frame_rate"])
    wav, sr = _load_audio(aligned[utt["id"]]["audio"])
    wav = _resample(wav, sr, sample_rate)
    return wav[: int(cut_sec * sample_rate)]


def export_bundle(
    paths: DataPaths,
    run_dir: str | Path,
    name: str,
    layers: int,
    checkpoint: str | Path | None = None,
    voice_wav: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    ckpt = Path(checkpoint) if checkpoint else latest_checkpoint(run_dir)
    out = paths.export / name
    out.mkdir(parents=True, exist_ok=True)
    model_config = paths.model_config(layers)
    step = export_weights(ckpt, model_config, out / "model.safetensors")
    shutil.copy(paths.tokenizer, out / "tokenizer.json")
    shutil.copy(paths.lexicon, out / "ta_marbuta_lexicon.json")
    cfg = bundle_config(model_config)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    sample_rate = int(cfg["mimi"]["sample_rate"])
    if voice_wav is not None:
        wav, sr = _load_audio(str(voice_wav))
        wav = _resample(wav, sr, sample_rate)
    else:
        wav = _default_voice_audio(paths, sample_rate)
    soundfile.write(str(out / "voice.wav"), wav, sample_rate)

    from egypocket.inference import EgyPocketTTS

    tts = EgyPocketTTS(out)
    tts.export_voice(out / "voice.wav", out / "voice.safetensors")
    info = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "step": step,
        "layers": layers,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out
