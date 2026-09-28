"""Step 8: precompute everything the trainer needs into one in-RAM cache per split.

pocket-tts's latent mode still reads audio and runs the Mimi encoder on every
training step: the first `stitch_frames` latents after the voice-prompt cut are
re-encoded from a cold encoder state (that is what the model has to produce at
inference, where nothing precedes the generated audio), and only the rest comes
from the stored full-utterance latents.

Cut points are word boundaries and are known in advance, so we precompute the
cold "stitch" latents for EVERY eligible cut. Training then reads nothing but
numpy arrays: no audio decoding, no resampling, no Mimi on the training GPU.

Layout of cache/<split>/:
    latents.npy   float32 [total_frames, 32]    full-utterance Mimi latents, concatenated
    stitch.npy    float32 [num_cuts, S, 32]     cold-start latents at each cut
    index.jsonl   one row per utterance: offset, frames, duration, words, cuts
    meta.json     stitch_frames S, frame rate, Mimi hash, cut rules, statistics

The cut rules replicate pocket-tts's DataLoader._choose_cut exactly.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import torch
from tqdm import tqdm

from egypocket.data.align import _load_audio, _resample, quality_filter
from egypocket.data.jsonl import read_json, read_jsonl, write_json, write_jsonl

logger = logging.getLogger(__name__)

MIN_CUT_SEC = 1.0  # pocket-tts DataLoader.MIN_CUT_SEC
TRAIL_SEC = 0.2  # pocket-tts DataLoader.TRAIL_SEC
CALIBRATION_MARGIN_FRAMES = 4


@dataclass
class CacheMeta:
    stitch_frames: int
    frame_rate: float
    sample_rate: int
    mimi_hash: str
    max_voice_prompt_sec: float
    min_cut_sec: float
    num_utterances: int
    num_frames: int
    num_cuts: int
    hours: float


def mimi_encode_hash(mimi: torch.nn.Module) -> str:
    """Hash of the weights on Mimi's encode path (as pocket-tts precompute_latents)."""
    h = hashlib.sha256()
    for name in ("encoder", "encoder_transformer", "downsample"):
        module = getattr(mimi, name, None)
        if module is None:
            continue
        for k, v in sorted(module.state_dict().items()):
            h.update(f"{name}.{k}:{tuple(v.shape)}".encode())
            h.update(v.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()


def load_mimi(model_config: str | Path, device: torch.device):
    """Frozen Mimi codec from the (gated) weights bundle referenced by a pocket config."""
    from pocket_tts.models.mimi import build_mimi
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary

    config = load_config(Path(model_config))
    if config.weights_path is None:
        raise ValueError(f"{model_config} has no weights_path")
    mimi = build_mimi(config.mimi)
    state = safetensors.torch.load_file(str(download_if_necessary(str(config.weights_path))))
    mimi_state = {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}
    mimi.load_state_dict(mimi_state, strict=True)
    enc_max = max(v.abs().max().item() for k, v in mimi_state.items() if k.startswith("encoder."))
    if enc_max == 0:
        raise SystemExit(
            f"{config.weights_path} ships an all-zero Mimi encoder (the release without voice "
            "cloning). Accept the terms of https://huggingface.co/kyutai/pocket-tts and log in "
            "with an HF token so the voice-cloning weights can be downloaded."
        )
    mimi.eval().to(device)
    for p in mimi.parameters():
        p.requires_grad_(False)
    # Streaming decode looks its state up by module name (pocket-tts stamp_state_names).
    from pocket_tts.modules.stateful_module import StatefulModule

    for name, module in mimi.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name
    return mimi


def eligible_cuts(
    words: list[dict[str, Any]], duration: float, max_voice_prompt_sec: float
) -> list[tuple[float, int]]:
    """(cut seconds, index of the first target word) — pocket-tts _choose_cut rules."""
    cuts = []
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        if prev["end"] is None or cur["start"] is None:
            continue
        cut = 0.5 * (prev["end"] + cur["start"])
        if cut >= MIN_CUT_SEC and duration - cut >= MIN_CUT_SEC:
            cuts.append((cut, i))
    if not cuts:
        return []
    window = max_voice_prompt_sec if max_voice_prompt_sec > 0 else float("inf")
    eligible = [(c, i) for c, i in cuts if words[i - 1]["end"] < window]
    return eligible or cuts[:1]


@torch.no_grad()
def measure_stitch_frames(mimi, audio: torch.Tensor) -> tuple[int, float]:
    """Frames after a cut where cold and warm encodings differ (pocket-tts calibration)."""
    fs = mimi.frame_size
    full = mimi.encode_to_latent(audio)
    k = full.shape[1] // 2
    cold = mimi.encode_to_latent(audio[..., k * fs :])
    n = min(cold.shape[1], full.shape[1] - k) - 1
    rel = (cold[:, :n] - full[:, k : k + n]).norm(dim=-1) / (full[:, k : k + n].norm(dim=-1) + 1e-8)
    rel = rel.max(dim=0).values
    prompt_cold = mimi.encode_to_latent(audio[..., : k * fs])
    pn = min(prompt_cold.shape[1], k) - 1
    floor = (
        ((prompt_cold[:, :pn] - full[:, :pn]).norm(dim=-1) / (full[:, :pn].norm(dim=-1) + 1e-8))
        .max()
        .item()
    )
    above = (rel > max(3 * floor, 1e-3)).nonzero()
    frames = int(above.max().item()) + 1 if above.numel() else 0
    return frames + CALIBRATION_MARGIN_FRAMES, floor


def _load_24k(path: str, sample_rate: int) -> np.ndarray:
    wav, sr = _load_audio(path)
    return _resample(wav, sr, sample_rate)


def _pad_batch(wavs: list[np.ndarray], multiple: int) -> torch.Tensor:
    length = max(len(w) for w in wavs)
    length = ((length + multiple - 1) // multiple) * multiple
    out = torch.zeros(len(wavs), 1, length)
    for b, w in enumerate(wavs):
        out[b, 0, : len(w)] = torch.from_numpy(w)
    return out


def calibrate_stitch_frames(
    rows: list[dict[str, Any]], mimi, device, n: int = 8, max_sec: float = 12.0
) -> tuple[int, float]:
    """Measure on n long clips (cut to max_sec: the cold/warm gap closes within ~2 s)."""
    longest = sorted(rows, key=lambda r: -r["duration"])[:n]
    wavs = [_load_24k(r["audio"], mimi.sample_rate) for r in longest]
    length = min(min(len(w) for w in wavs), int(max_sec * mimi.sample_rate))
    length -= length % mimi.frame_size
    audio = torch.stack([torch.from_numpy(w[:length]) for w in wavs])[:, None].to(device)
    return measure_stitch_frames(mimi, audio)


def build_cache(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    mimi,
    device: torch.device,
    stitch_frames: int,
    max_voice_prompt_sec: float = 5.0,
    batch_size: int = 16,
    stitch_batch_size: int = 128,
) -> CacheMeta:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr, fr, fs = mimi.sample_rate, float(mimi.frame_rate), mimi.frame_size
    S = stitch_frames
    rows = sorted(rows, key=lambda r: r["duration"])

    latents_parts: list[np.ndarray] = []
    stitch_parts: list[np.ndarray] = []
    index: list[dict[str, Any]] = []
    offset = 0
    n_cuts = 0
    skipped_no_cut = 0
    pending_stitch: list[np.ndarray] = []

    def flush_stitch():
        nonlocal pending_stitch
        if not pending_stitch:
            return
        for s0 in range(0, len(pending_stitch), stitch_batch_size):
            chunk = pending_stitch[s0 : s0 + stitch_batch_size]
            audio = torch.from_numpy(np.stack(chunk))[:, None].to(device)
            lat = mimi.encode_to_latent(audio)[:, :S].float().cpu().numpy()
            stitch_parts.append(lat)
        pending_stitch = []

    for b0 in tqdm(range(0, len(rows), batch_size), desc=f"encode {out_dir.name}"):
        batch = rows[b0 : b0 + batch_size]
        wavs = [_load_24k(r["audio"], sr) for r in batch]
        with torch.no_grad():
            full = mimi.encode_to_latent(_pad_batch(wavs, fs).to(device)).float().cpu().numpy()
        for row, wav, lat in zip(batch, wavs, full, strict=True):
            frames = min(max(1, int(len(wav) * fr / sr)), lat.shape[0])
            cuts = []
            seen = set()
            for cut_sec, word_idx in eligible_cuts(row["words"], row["duration"], max_voice_prompt_sec):
                cut_frame = min(max(round(cut_sec * fr), 1), frames - 1)
                if cut_frame in seen:
                    continue
                seen.add(cut_frame)
                window = np.zeros(S * fs, dtype=np.float32)
                piece = wav[cut_frame * fs : (cut_frame + S) * fs]
                window[: len(piece)] = piece
                pending_stitch.append(window)
                stitch_len = min(S, frames - cut_frame)
                cuts.append([cut_frame, word_idx, n_cuts, stitch_len])
                n_cuts += 1
            if not cuts:
                skipped_no_cut += 1
                continue
            latents_parts.append(lat[:frames].astype(np.float32))
            index.append(
                {
                    "id": row["id"],
                    "offset": offset,
                    "frames": frames,
                    "duration": row["duration"],
                    "words": [
                        {"word": w["word"], "diac": w["diac"], "start": w["start"], "end": w["end"]}
                        for w in row["words"]
                    ],
                    "cuts": cuts,
                }
            )
            offset += frames
        if len(pending_stitch) >= stitch_batch_size:
            with torch.no_grad():
                flush_stitch()
    with torch.no_grad():
        flush_stitch()

    latents = np.concatenate(latents_parts, axis=0)
    stitch = np.concatenate(stitch_parts, axis=0) if stitch_parts else np.zeros((0, S, 32), np.float32)
    assert stitch.shape[0] == n_cuts, (stitch.shape, n_cuts)
    np.save(out_dir / "latents.npy", latents)
    np.save(out_dir / "stitch.npy", stitch)
    write_jsonl(out_dir / "index.jsonl", index)
    meta = CacheMeta(
        stitch_frames=S,
        frame_rate=fr,
        sample_rate=sr,
        mimi_hash=mimi_encode_hash(mimi),
        max_voice_prompt_sec=max_voice_prompt_sec,
        min_cut_sec=MIN_CUT_SEC,
        num_utterances=len(index),
        num_frames=int(latents.shape[0]),
        num_cuts=n_cuts,
        hours=round(sum(r["duration"] for r in index) / 3600, 3),
    )
    write_json(out_dir / "meta.json", meta.__dict__)
    logger.info(
        f"{out_dir}: {meta.num_utterances} utterances ({meta.hours} h), {n_cuts} cuts, "
        f"{skipped_no_cut} skipped (no valid cut), stitch_frames={S}"
    )
    return meta


def ctc_threshold(rows: list[dict[str, Any]], drop_worst_fraction: float) -> float:
    """Score below which the worst `drop_worst_fraction` of clips fall.

    The CTC score is the mean per-character log-probability under an MSA
    wav2vec2 model, so its absolute level depends on the dialect (a median
    around -1.8 on this corpus); a relative cut-off is robust to that.
    """
    scores = np.array([r["ctc_score"] for r in rows], dtype=np.float64)
    if drop_worst_fraction <= 0 or len(scores) == 0:
        return float("-inf")
    return float(np.quantile(scores, drop_worst_fraction))


def build_all_caches(
    paths,
    model_config: str | Path,
    drop_worst_fraction: float = 0.03,
    min_ctc_score: float | None = None,
    min_timed_ratio: float = 0.9,
    max_voice_prompt_sec: float = 5.0,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    """Build train/valid caches. Clips are dropped when their CTC score is below
    `min_ctc_score`, or (default) below the train split's `drop_worst_fraction` quantile."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    all_train = read_jsonl(paths.aligned("train"))
    if min_ctc_score is None:
        min_ctc_score = ctc_threshold(all_train, drop_worst_fraction)
    logger.info(f"CTC score cut-off: {min_ctc_score:.3f}")
    mimi = load_mimi(model_config, dev)
    train_rows = quality_filter(all_train, min_ctc_score, min_timed_ratio)
    stitch_frames, floor = calibrate_stitch_frames(train_rows, mimi, dev)
    logger.info(f"stitch_frames={stitch_frames} (noise floor {floor:.1e})")
    summary: dict[str, Any] = {"stitch_frames": stitch_frames, "min_ctc_score": min_ctc_score}
    for split in ("train", "valid"):
        rows = read_jsonl(paths.aligned(split))
        kept = quality_filter(rows, min_ctc_score, min_timed_ratio)
        meta = build_cache(
            kept,
            paths.cache_dir(split),
            mimi,
            dev,
            stitch_frames,
            max_voice_prompt_sec=max_voice_prompt_sec,
            batch_size=batch_size,
        )
        summary[split] = {"aligned": len(rows), "after_quality_filter": len(kept), **meta.__dict__}
    write_json(paths.cache / "summary.json", summary)
    return summary


def read_meta(cache_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(cache_dir) / "meta.json")
