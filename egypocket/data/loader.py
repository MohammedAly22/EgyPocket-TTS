"""Training batches straight from the in-RAM latent cache (see cache.py).

Per sample (same semantics as pocket-tts's latent-mode DataLoader):
  * pick one of the utterance's precomputed word-boundary cuts at random;
  * voice prompt = the utterance's latents before the cut (capped at
    max_voice_prompt_sec, then 0..prompt_trim_max_sec trimmed off its end);
  * target = cold-start stitch latents + stored latents after them, up to the
    last word's end + 0.2 s (so EOS has a consistent target);
  * text = the words after the cut, each word plain or diacritized according to
    DiacriticArgs; the final "." is dropped with probability final_punct_dropout.
The only per-step CPU work is slicing arrays and tokenizing a short string.
"""

import logging
import queue
import random
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from egypocket.data.cache import TRAIL_SEC
from egypocket.data.jsonl import read_json, read_jsonl
from egypocket.text.frontend import subsample_marks

logger = logging.getLogger(__name__)


class LatentCache:
    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        if not (cache_dir / "meta.json").exists():
            raise FileNotFoundError(f"no latent cache at {cache_dir}: run the cache step first")
        self.dir = cache_dir
        self.meta: dict[str, Any] = read_json(cache_dir / "meta.json")
        self.latents = np.load(cache_dir / "latents.npy")
        self.stitch = np.load(cache_dir / "stitch.npy")
        self.utts: list[dict[str, Any]] = read_jsonl(cache_dir / "index.jsonl")
        self.frame_rate = float(self.meta["frame_rate"])
        self.stitch_frames = int(self.meta["stitch_frames"])

    def __len__(self) -> int:
        return len(self.utts)

    def utterance_latents(self, u: int) -> np.ndarray:
        utt = self.utts[u]
        return self.latents[utt["offset"] : utt["offset"] + utt["frames"]]


@dataclass
class Batch:
    latents: torch.Tensor  # [B, T, C] target latents (raw Mimi latent space)
    mask: torch.Tensor  # [B, T] valid target frames
    text_tokens: list[torch.Tensor]  # ragged [L_b] token ids
    texts: list[str]
    prompt_latents: torch.Tensor  # [B, P, C]
    num_prompt_frames: torch.Tensor  # [B]


def build_text(
    words: list[dict[str, Any]], diacritics: Any | None, rng: random.Random
) -> str:
    """Join target words, choosing plain or diacritized spelling per word."""
    have_diac = diacritics is not None and any(w["diac"] for w in words)
    mode = "plain"
    if have_diac:
        r = rng.random()
        if r < diacritics.p_full:
            mode = "full"
        elif r < diacritics.p_full + diacritics.p_partial:
            mode = "partial"
    word_prob = rng.uniform(*diacritics.partial_word_prob) if mode == "partial" else 0.0
    out = []
    for w in words:
        use = w["diac"] is not None and (
            mode == "full" or (mode == "partial" and rng.random() < word_prob)
        )
        if not use:
            out.append(w["word"])
            continue
        token = w["diac"]
        if rng.random() < diacritics.sparse_word_prob:
            token = subsample_marks(token, rng.uniform(*diacritics.sparse_keep_prob), rng)
        out.append(token)
    return " ".join(out)


class CachedBatches:
    def __init__(
        self,
        cache: LatentCache,
        encode: Callable[[str], list[int]],
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        max_duration_sec: float = 20.0,
        max_voice_prompt_sec: float = 5.0,
        prompt_trim_max_sec: float = 0.0,
        final_punct_dropout: float = 0.0,
        num_bucket_batches: int = 1,
        diacritics: Any | None = None,
        prefetch: int = 8,
    ):
        self.cache = cache
        self.encode = encode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_duration_sec = max_duration_sec
        self.max_voice_prompt_sec = max_voice_prompt_sec
        self.prompt_trim_max_sec = prompt_trim_max_sec
        self.final_punct_dropout = final_punct_dropout
        self.num_bucket_batches = max(1, num_bucket_batches)
        self.diacritics = diacritics
        self.prefetch = prefetch
        self.rng = random.Random(seed)
        self.indices = list(range(rank, len(cache), world_size))
        if len(self.indices) < batch_size and shuffle:
            raise ValueError(
                f"{len(self.indices)} utterances for this rank but batch_size={batch_size}"
            )
        cached_window = float(cache.meta["max_voice_prompt_sec"])
        if abs(cached_window - max_voice_prompt_sec) > 1e-6:
            raise ValueError(
                f"cache {cache.dir} was built with max_voice_prompt_sec={cached_window}, "
                f"config asks for {max_voice_prompt_sec}: rebuild the cache or fix the config"
            )

    # ------------------------------------------------------------------ samples
    def _trim_prompt(self, prompt: np.ndarray) -> np.ndarray:
        if self.prompt_trim_max_sec <= 0:
            return prompt
        fr = self.cache.frame_rate
        drop = self.rng.randint(0, int(round(self.prompt_trim_max_sec * fr)))
        keep = max(prompt.shape[0] - drop, min(prompt.shape[0], int(fr)))
        return prompt[:keep]

    def sample(self, u: int) -> tuple[list[int], str, np.ndarray, np.ndarray]:
        cache = self.cache
        utt = cache.utts[u]
        fr = cache.frame_rate
        lat = cache.utterance_latents(u)
        stored = lat.shape[0]
        cut_frame, word_idx, stitch_row, stitch_len = self.rng.choice(utt["cuts"])

        text = build_text(utt["words"][word_idx:], self.diacritics, self.rng)
        if self.final_punct_dropout > 0 and self.rng.random() < self.final_punct_dropout:
            text = re.sub(r"\.\s*$", "", text) or text
        tokens = self.encode(text)

        cut_sec = cut_frame / fr
        end = utt["duration"]
        ends = [w["end"] for w in utt["words"] if w["end"] is not None]
        last = max(ends) if ends else None
        if last is not None and last > cut_sec:
            end = min(utt["duration"], last + TRAIL_SEC)
        target_frames = int(min(end - cut_sec, self.max_duration_sec) * fr)
        target_frames = max(1, min(target_frames, stored - cut_frame))
        n_stitch = min(stitch_len, target_frames)
        target = np.concatenate(
            [
                cache.stitch[stitch_row, :n_stitch],
                lat[cut_frame + n_stitch : cut_frame + target_frames],
            ],
            axis=0,
        )

        cap = max(1, int(self.max_voice_prompt_sec * fr)) if self.max_voice_prompt_sec > 0 else stored
        prompt = self._trim_prompt(lat[: min(cut_frame, cap)])
        return tokens, text, prompt, target

    # ------------------------------------------------------------------ batches
    @staticmethod
    def _row_len(sample: tuple[list[int], str, np.ndarray, np.ndarray]) -> int:
        tokens, _, prompt, target = sample
        return len(tokens) + prompt.shape[0] + target.shape[0]

    @staticmethod
    def collate(samples: list[tuple[list[int], str, np.ndarray, np.ndarray]]) -> Batch:
        B = len(samples)
        C = samples[0][3].shape[-1]
        T = max(s[3].shape[0] for s in samples)
        P = max(1, max(s[2].shape[0] for s in samples))
        latents = torch.zeros(B, T, C)
        mask = torch.zeros(B, T, dtype=torch.bool)
        prompts = torch.zeros(B, P, C)
        num_prompt = torch.zeros(B, dtype=torch.long)
        for b, (_, _, prompt, target) in enumerate(samples):
            latents[b, : target.shape[0]] = torch.from_numpy(target)
            mask[b, : target.shape[0]] = True
            if prompt.shape[0]:
                prompts[b, : prompt.shape[0]] = torch.from_numpy(prompt)
            num_prompt[b] = max(1, prompt.shape[0])
        tokens = [torch.tensor(s[0], dtype=torch.long) for s in samples]
        texts = [s[1] for s in samples]
        return Batch(latents, mask, tokens, texts, prompts, num_prompt)

    def _batches(self) -> Iterator[Batch]:
        while True:
            order = list(self.indices)
            if self.shuffle:
                self.rng.shuffle(order)
            pool_size = self.num_bucket_batches * self.batch_size
            pool: list[tuple[list[int], str, np.ndarray, np.ndarray]] = []
            for pos, u in enumerate(order):
                pool.append(self.sample(u))
                last = pos == len(order) - 1
                if len(pool) < pool_size and not last:
                    continue
                pool.sort(key=self._row_len)
                n_full = len(pool) // self.batch_size
                batches = [pool[i * self.batch_size : (i + 1) * self.batch_size] for i in range(n_full)]
                rest = pool[n_full * self.batch_size :]
                if last and rest and not self.shuffle:
                    batches.append(rest)  # evaluation: keep every utterance
                    rest = []
                pool = rest
                if self.shuffle:
                    self.rng.shuffle(batches)
                for batch in batches:
                    yield self.collate(batch)
            if not self.shuffle:
                return

    def __iter__(self) -> Iterator[Batch]:
        """Batches are produced by a background thread and pinned for fast H2D copies."""
        q: queue.Queue[Batch | BaseException | None] = queue.Queue(maxsize=self.prefetch)
        pin = torch.cuda.is_available()

        def worker():
            try:
                for batch in self._batches():
                    if pin:
                        batch.latents = batch.latents.pin_memory()
                        batch.prompt_latents = batch.prompt_latents.pin_memory()
                    q.put(batch)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
                q.put(exc)
            q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
