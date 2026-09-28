"""CPU inference for EgyPocket-TTS.

    tts = EgyPocketTTS("export/egypocket_6l")
    audio = tts.generate("النهارده الجو حلو قوي.")            # numpy float32, 24 kHz
    for chunk in tts.stream("..."): ...                         # low-latency streaming
    tts.generate("رحت المدينةْ امبارح.")                        # diacritics override pronunciation

The heavy lifting is the stock pocket-tts `TTSModel` (streaming Mimi decoding
in a parallel thread, KV-cached 6-layer backbone, 1-step LSD sampling). This
module adds the Egyptian text frontend (identical to training), sentence
chunking and bundle loading.
"""

import json
import logging
import re
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from egypocket.text.chars import LETTERS_SET
from egypocket.text.frontend import normalize_text_verbose

logger = logging.getLogger(__name__)


def split_chunks(text: str, max_letters: int = 160) -> list[str]:
    """Split normalized text into sentence-level chunks of at most ~max_letters letters.

    The model was trained on utterances of up to ~18 s (~200 letters); longer
    input is generated chunk by chunk from the same voice state.
    """

    def letters(s: str) -> int:
        return sum(c in LETTERS_SET for c in s)

    def split_long(sentence: str) -> list[str]:
        if letters(sentence) <= max_letters:
            return [sentence]
        parts = [p.strip() for p in re.split(r"(?<=,)\s", sentence) if p.strip()]
        out: list[str] = []
        for part in parts:
            if letters(part) <= max_letters:
                out.append(part)
                continue
            words, cur = part.split(), []
            for w in words:
                if cur and letters(" ".join(cur + [w])) > max_letters:
                    out.append(" ".join(cur))
                    cur = []
                cur.append(w)
            if cur:
                out.append(" ".join(cur))
        return out

    sentences = [s.strip() for s in re.split(r"(?<=\.)\s", text) if s.strip()]
    pieces = [p for s in sentences for p in split_long(s)]
    chunks: list[str] = []
    for piece in pieces:
        if chunks and letters(chunks[-1]) + letters(piece) <= max_letters:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    # Every chunk ends a sentence, as in training.
    return [c if c.endswith(".") else c.rstrip(",") + "." for c in chunks]


@dataclass
class GenerationStats:
    audio_sec: float
    wall_sec: float
    first_chunk_sec: float

    @property
    def real_time_factor(self) -> float:
        """Seconds of audio produced per second of compute (>1 = faster than real time)."""
        return self.audio_sec / max(self.wall_sec, 1e-9)


class EgyPocketTTS:
    def __init__(
        self,
        bundle_dir: str | Path,
        temperature: float | None = None,
        eos_threshold: float = -4.0,
        quantize: bool = False,
        num_threads: int | None = None,
        max_letters_per_chunk: int = 160,
    ):
        from pocket_tts import TTSModel

        self.bundle = Path(bundle_dir).resolve()
        cfg = yaml.safe_load((self.bundle / "config.yaml").read_text(encoding="utf-8"))
        cfg["weights_path"] = str(self.bundle / cfg["weights_path"])
        lut = cfg["flow_lm"]["lookup_table"]
        lut["tokenizer_path"] = str(self.bundle / lut["tokenizer_path"])
        self._resolved = Path(tempfile.mkdtemp(prefix="egypocket_")) / "config.yaml"
        self._resolved.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        if num_threads:
            torch.set_num_threads(num_threads)
        self.model = TTSModel.load_model(
            config=str(self._resolved),
            temp=temperature,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )
        # Character tokens: ~13/s of speech plain, up to ~25/s fully diacritized.
        # The generation cap is token_count / estimate + 2 s, so 4 keeps >= 2.5x headroom.
        self.model._TOKENS_PER_SECOND_ESTIMATE = 4.0
        lexicon_path = self.bundle / "ta_marbuta_lexicon.json"
        self.lexicon = frozenset(json.loads(lexicon_path.read_text(encoding="utf-8")))
        self.sample_rate = self.model.sample_rate
        self.max_letters_per_chunk = max_letters_per_chunk
        self._voices: dict[str, dict] = {}
        self.last_stats: GenerationStats | None = None

    # ------------------------------------------------------------------ text
    def normalize(self, text: str) -> str:
        normalized, dropped = normalize_text_verbose(text, self.lexicon)
        if dropped:
            logger.warning(f"dropped unsupported characters: {''.join(sorted(set(dropped)))!r}")
        return normalized

    def chunks(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        if not normalized:
            raise ValueError("nothing to say after normalization")
        return split_chunks(normalized, self.max_letters_per_chunk)

    # ------------------------------------------------------------------ voice
    def voice_state(self, voice: str | Path | None = None) -> dict:
        """Voice prompt state: a .wav (cloned) or .safetensors (exported); default = bundle voice."""
        source = Path(voice) if voice is not None else self.bundle / "voice.safetensors"
        key = str(source.resolve())
        if key not in self._voices:
            self._voices[key] = self.model.get_state_for_audio_prompt(source)
        return self._voices[key]

    def export_voice(self, wav_path: str | Path, out_path: str | Path) -> Path:
        from pocket_tts import export_model_state

        state = self.model.get_state_for_audio_prompt(Path(wav_path))
        export_model_state(state, str(out_path))
        return Path(out_path)

    # ------------------------------------------------------------------ audio
    def stream(self, text: str, voice: str | Path | None = None) -> Iterator[np.ndarray]:
        """Yield audio chunks (float32, 24 kHz) as soon as they are decoded."""
        state = self.voice_state(voice)
        t0 = time.monotonic()
        first = None
        produced = 0
        for chunk_text in self.chunks(text):
            for audio in self.model.generate_audio_stream(
                state, chunk_text, max_tokens=100_000, copy_state=True
            ):
                if first is None:
                    first = time.monotonic() - t0
                wav = audio.detach().float().cpu().numpy()
                produced += wav.shape[-1]
                yield wav
        self.last_stats = GenerationStats(
            audio_sec=produced / self.sample_rate,
            wall_sec=time.monotonic() - t0,
            first_chunk_sec=first or 0.0,
        )

    def generate(self, text: str, voice: str | Path | None = None) -> np.ndarray:
        parts = list(self.stream(text, voice))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def save(self, text: str, out_path: str | Path, voice: str | Path | None = None) -> Path:
        import soundfile

        audio = self.generate(text, voice)
        soundfile.write(str(out_path), audio, self.sample_rate)
        return Path(out_path)
