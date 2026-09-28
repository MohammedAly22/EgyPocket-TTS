"""Step 6: word alignment, boundary refinement and pause punctuation.

Pocket TTS trains on utterances cut at a word boundary: audio before the cut
is the voice prompt, audio after the cut is the target and the words after the
cut are the text. That needs word timestamps, obtained here by CTC forced
alignment with an Arabic wav2vec2 model (batched Viterbi trellis adapted from
pocket-tts training/scripts/align_data.py, MIT licence, Kyutai).

On top of the raw CTC spikes:

* Boundaries are refined on the waveform: between two words we look for the
  longest silent run (energy far below the utterance's speech level) and put
  the boundary there, or at the energy minimum when there is no pause. Cuts
  therefore land in pauses, like the voice prompts used at inference.
* The corpus has no punctuation, so pauses become punctuation: a pause of at
  least `comma_sec` adds ",", at least `period_sec` adds "."; every utterance
  ends with ".". The model thus learns "," = short pause, "." = sentence end.
* Each utterance gets a CTC score (mean log-probability of its characters on
  the Viterbi path) used to drop clips whose transcript does not match.
"""

import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile
import torch
from tqdm import tqdm

from egypocket.data.jsonl import read_jsonl, write_json, write_jsonl
from egypocket.paths import SPLITS, DataPaths
from egypocket.text.chars import COMMA, PERIOD

logger = logging.getLogger(__name__)

DEFAULT_ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"


@dataclass
class PauseConfig:
    comma_sec: float = 0.22
    period_sec: float = 0.55
    min_silence_sec: float = 0.06  # shortest run that counts as a pause for boundary placement
    energy_hop_sec: float = 0.01
    speech_margin_db: float = 35.0  # silence = below (speech level - margin) ...
    floor_margin_db: float = 8.0  # ... or not far above the noise floor


def batched_viterbi(
    emissions: torch.Tensor,  # [B, Tmax, V] log-probs
    T: torch.Tensor,  # [B] valid frame counts
    token_lists: list[list[int]],
    blank: int,
) -> list[tuple[list[int], list[float]] | None]:
    """Per item: (frame of each token, log-prob of each token), or None if unalignable."""
    device = emissions.device
    B, Tmax, _ = emissions.shape
    N = torch.tensor([len(t) for t in token_lists], device=device)
    Nmax = max(1, int(N.max()))
    tok = torch.zeros(B, Nmax, dtype=torch.long, device=device)
    for b, t in enumerate(token_lists):
        if t:
            tok[b, : len(t)] = torch.tensor(t, device=device)
    neg = float("-inf")
    trellis = torch.full((Tmax + 1, B, Nmax + 1), neg, device=device)
    trellis[0, :, 0] = 0.0
    blank_em = emissions[:, :, blank]
    tok_em = emissions.gather(2, tok.unsqueeze(1).expand(B, Tmax, Nmax))
    for t in range(Tmax):
        prev = trellis[t]
        stay = prev + blank_em[:, t : t + 1]
        move = torch.cat([torch.full((B, 1), neg, device=device), prev[:, :-1] + tok_em[:, t]], 1)
        new = torch.maximum(stay, move)
        trellis[t + 1] = torch.where((t < T).view(B, 1), new, prev)

    tr_all = trellis.permute(1, 0, 2).cpu().numpy()
    blank_all = blank_em.float().cpu().numpy()
    tok_all = tok_em.float().cpu().numpy()
    results: list[tuple[list[int], list[float]] | None] = []
    for b in range(B):
        n, t_end = int(N[b]), int(T[b])
        tr = tr_all[b]
        if n == 0 or t_end < n or not np.isfinite(tr[t_end, n]):
            results.append(None)
            continue
        frames = [0] * n
        j = n
        for t in range(t_end, 0, -1):
            if j == 0:
                break
            stay = tr[t - 1, j] + blank_all[b, t - 1]
            move = tr[t - 1, j - 1] + tok_all[b, t - 1, j - 1]
            if move >= stay:
                j -= 1
                frames[j] = t - 1
        if j != 0:
            results.append(None)
            continue
        scores = [float(tok_all[b, f, k]) for k, f in enumerate(frames)]
        results.append((frames, scores))
    return results


def energy_db(wav: np.ndarray, sr: int, hop_sec: float) -> np.ndarray:
    hop = max(1, int(sr * hop_sec))
    n = len(wav) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = wav[: n * hop].reshape(n, hop).astype(np.float64)
    db = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
    kernel = np.ones(3) / 3.0
    return np.convolve(db, kernel, mode="same").astype(np.float32)


def silence_threshold(db: np.ndarray, cfg: PauseConfig) -> float:
    speech = float(np.percentile(db, 90))
    floor = float(np.percentile(db, 5))
    return max(speech - cfg.speech_margin_db, min(floor + cfg.floor_margin_db, speech - 15.0))


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, end) of the longest True run in mask, end exclusive; (0, 0) if none."""
    best, best_start, start = 0, 0, None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, best_start = i - start, start
            start = None
    return best_start, best_start + best


def refine_words(
    words: list[dict[str, Any]], wav: np.ndarray, sr: int, duration: float, cfg: PauseConfig
) -> None:
    """Move word boundaries into pauses and attach pause punctuation, in place."""
    hop = cfg.energy_hop_sec
    db = energy_db(wav, sr, hop)
    if len(db) == 0:
        return
    silent = db < silence_threshold(db, cfg)
    n = len(db)
    min_run = max(1, round(cfg.min_silence_sec / hop))

    for k in range(len(words) - 1):
        cur, nxt = words[k], words[k + 1]
        if cur["end"] is None or nxt["start"] is None:
            continue
        lo = max(0, int(cur["end"] / hop) - 2)
        hi = min(n, int(np.ceil(nxt["start"] / hop)) + 2)
        if hi - lo < 1:
            mid = round(0.5 * (cur["end"] + nxt["start"]), 3)
            cur["end"] = nxt["start"] = mid
            continue
        s, e = _longest_run(silent[lo:hi])
        if e - s >= min_run:
            start_sec, end_sec = (lo + s) * hop, (lo + e) * hop
            cur["end"], nxt["start"] = round(start_sec, 3), round(end_sec, 3)
            pause = end_sec - start_sec
            if pause >= cfg.period_sec:
                cur["punct"] = PERIOD
            elif pause >= cfg.comma_sec:
                cur["punct"] = COMMA
        else:
            cut = (lo + int(np.argmin(db[lo:hi]))) * hop + 0.5 * hop
            cur["end"] = nxt["start"] = round(cut, 3)

    last = words[-1]
    if last["end"] is not None:
        # CTC spikes stop before the last phone ends: extend to the speech offset.
        i = min(n, int(last["end"] / hop))
        while i < n and not silent[i : i + min_run].all():
            i += 1
        last["end"] = round(min(duration, max(last["end"], i * hop)), 3)
    first = words[0]
    if first["start"] is not None:
        i = min(n - 1, int(first["start"] / hop))
        while i > 0 and not silent[max(0, i - min_run) : i].all():
            i -= 1
        first["start"] = round(max(0.0, min(first["start"], i * hop)), 3)


class Aligner:
    def __init__(self, model_name: str = DEFAULT_ALIGN_MODEL, device: str | None = None):
        import transformers
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        transformers.logging.set_verbosity_error()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device).eval()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model.to(self.dtype)
        self.vocab: dict[str, int] = processor.tokenizer.get_vocab()
        self.blank = processor.tokenizer.pad_token_id
        self.delim = self.vocab[processor.tokenizer.word_delimiter_token]
        fe = processor.feature_extractor
        self.sr = fe.sampling_rate
        self.normalize = bool(getattr(fe, "do_normalize", True))
        self.use_attention_mask = bool(getattr(fe, "return_attention_mask", True))

    def tokens(self, words: list[str]) -> tuple[list[int], list[int]]:
        toks, word_of = [], []
        first = True
        for w_idx, w in enumerate(words):
            ids = [self.vocab[c] for c in w if c in self.vocab and c != "|"]
            if not ids:
                continue
            if not first:
                toks.append(self.delim)
                word_of.append(-1)
            first = False
            toks.extend(ids)
            word_of.extend([w_idx] * len(ids))
        return toks, word_of

    @torch.no_grad()
    def emissions(self, wavs16: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        lens = [len(w) for w in wavs16]
        x = torch.zeros(len(wavs16), max(lens))
        for b, w in enumerate(wavs16):
            t = torch.from_numpy(w)
            if self.normalize:
                t = (t - t.mean()) / torch.sqrt(t.var() + 1e-7)
            x[b, : len(w)] = t
        attn = (torch.arange(max(lens))[None, :] < torch.tensor(lens)[:, None]).long()
        logits = self.model(
            x.to(self.device, self.dtype),
            attention_mask=attn.to(self.device) if self.use_attention_mask else None,
        ).logits.float()
        T = torch.tensor(
            [int(self.model._get_feat_extract_output_lengths(n)) for n in lens], device=self.device
        )
        return logits.log_softmax(-1), T


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    wav, sr = soundfile.read(path, dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return wav
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target)
    return resample_poly(wav, target // g, sr // g).astype(np.float32)


def _prefetch(rows: list[dict[str, Any]], depth: int = 64) -> Iterator[tuple[dict[str, Any], Any]]:
    q: queue.Queue[tuple[dict[str, Any], Any] | None] = queue.Queue(maxsize=depth)

    def work():
        for row in rows:
            try:
                q.put((row, _load_audio(row["audio"])))
            except Exception as exc:  # noqa: BLE001 - reported per row
                q.put((row, exc))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def align_split(
    rows: list[dict[str, Any]],
    diac: dict[str, str | None],
    aligner: Aligner,
    cfg: PauseConfig,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = sorted(rows, key=lambda r: r["duration"])  # less padding per batch
    out: list[dict[str, Any]] = []
    stats = {"aligned": 0, "failed": 0, "unreadable": 0, "diac_mismatch": 0}
    batch: list[tuple[dict[str, Any], np.ndarray, int]] = []

    def flush():
        wavs16 = [_resample(w, sr, aligner.sr) for _, w, sr in batch]
        emissions, T = aligner.emissions(wavs16)
        token_info = [aligner.tokens(r["text"].split()) for r, _, _ in batch]
        paths = batched_viterbi(emissions, T, [t for t, _ in token_info], aligner.blank)
        for (row, wav, sr), (_, word_of), path, t_frames, w16 in zip(
            batch, token_info, paths, T.tolist(), wavs16, strict=True
        ):
            words_txt = row["text"].split()
            if path is None:
                stats["failed"] += 1
                continue
            frames, scores = path
            sec_per_frame = (len(w16) / aligner.sr) / t_frames
            spans: dict[int, tuple[int, int]] = {}
            word_scores = []
            for f, w_idx, sc in zip(frames, word_of, scores, strict=True):
                if w_idx < 0:
                    continue
                s, e = spans.get(w_idx, (f, f))
                spans[w_idx] = (min(s, f), max(e, f))
                word_scores.append(sc)
            d_words = (diac.get(row["id"]) or "").split()
            if d_words and len(d_words) != len(words_txt):
                stats["diac_mismatch"] += 1
                d_words = []
            words = []
            for i, w in enumerate(words_txt):
                span = spans.get(i)
                words.append(
                    {
                        "word": w,
                        "diac": d_words[i] if d_words else None,
                        "start": round(span[0] * sec_per_frame, 3) if span else None,
                        "end": round((span[1] + 1) * sec_per_frame, 3) if span else None,
                    }
                )
            refine_words(words, wav, sr, row["duration"], cfg)
            for i, w in enumerate(words):
                punct = w.pop("punct", "")
                if i == len(words) - 1:
                    punct = PERIOD  # every utterance ends a sentence
                w["word"] += punct
                if w["diac"] is not None:
                    w["diac"] += punct
            timed = sum(w["start"] is not None for w in words)
            out.append(
                {
                    "id": row["id"],
                    "audio": row["audio"],
                    "duration": row["duration"],
                    "split": row["split"],
                    "text": " ".join(w["word"] for w in words),
                    "words": words,
                    "ctc_score": round(float(np.mean(word_scores)), 4) if word_scores else -99.0,
                    "timed_ratio": round(timed / len(words), 4),
                }
            )
            stats["aligned"] += 1
        batch.clear()

    for row, loaded in tqdm(_prefetch(rows), total=len(rows), desc="align"):
        if isinstance(loaded, Exception):
            stats["unreadable"] += 1
            continue
        wav, sr = loaded
        batch.append((row, wav, sr))
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    out.sort(key=lambda r: r["id"])
    return out, stats


def align_all(
    paths: DataPaths,
    model_name: str = DEFAULT_ALIGN_MODEL,
    device: str | None = None,
    batch_size: int = 16,
    cfg: PauseConfig | None = None,
) -> dict[str, dict[str, int]]:
    cfg = cfg or PauseConfig()
    diac = {r["id"]: r["diac"] for r in read_jsonl(paths.catt(2))}
    aligner = Aligner(model_name, device)
    all_stats = {}
    for split in SPLITS.values():
        rows = read_jsonl(paths.restored(split))
        aligned, stats = align_split(rows, diac, aligner, cfg, batch_size)
        write_jsonl(paths.aligned(split), aligned)
        all_stats[split] = stats
        logger.info(f"{split}: {stats}")
    write_json(paths.work / "align_stats.json", all_stats)
    return all_stats


def quality_filter(
    rows: list[dict[str, Any]], min_ctc_score: float, min_timed_ratio: float = 0.9
) -> list[dict[str, Any]]:
    return [
        r for r in rows if r["ctc_score"] >= min_ctc_score and r["timed_ratio"] >= min_timed_ratio
    ]
