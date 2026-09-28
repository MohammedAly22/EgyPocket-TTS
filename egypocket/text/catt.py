"""CATT (Egyptian-Arabic diacritizer) integration — used for DATA PREPARATION only.

CATT is never part of inference. It annotates the training transcripts so the
model learns what diacritics mean; at inference the user writes diacritics only
where the model reads a word wrongly.

CATT emits special symbols besides the standard tashkeel:
    ^  <  >   (tafkhim/tarqiq and imala marks)  -> removed, never reach training text
    ؞         (mothalatha, e.g. Egyptian ق read as a glottal stop) -> "~"
`normalize_catt_output` is the deterministic function that applies this.
"""

import importlib
import logging
import sys
import types
import unicodedata
from pathlib import Path

from egypocket.text.chars import CATT_MOTHALATHA, CATT_REMOVED, EGY_MARKER
from egypocket.text.frontend import canonicalize_diacritics, strip_diacritics

logger = logging.getLogger(__name__)

# ASCII symbols plus their full-width look-alikes, all removed.
_REMOVE = {ord(c): None for c in CATT_REMOVED + ("＾", "＜", "＞")}
_REMOVE[ord(CATT_MOTHALATHA)] = EGY_MARKER


def normalize_catt_output(text: str) -> str:
    """Clean a CATT output into EgyPocket training text.

    1. NFC, so precomposed and decomposed inputs behave the same.
    2. Remove ^ < > entirely. Diacritics that followed one of them now follow
       the letter itself (e.g. "ت^َّ" -> "تَّ"), which is where they belong.
    3. Replace ؞ with ~.
    4. Canonical diacritic order per letter ([~][shadda][vowel]); diacritics
       are PRESERVED, never stripped.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVE)
    return canonicalize_diacritics(text)


def _import_catt_modules(catt_dir: Path) -> tuple[types.ModuleType, types.ModuleType]:
    """Import CATT's tokenizer/model modules without running the package __init__.

    The package __init__ also imports the MSA/ONNX variants, whose weights are
    not needed (and not shipped); registering an empty package object with the
    right __path__ lets the relative imports inside model.py still work.
    """
    name = "catt_tashkeel"
    existing = sys.modules.get(name)
    if existing is None or list(getattr(existing, "__path__", [])) != [str(catt_dir)]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(catt_dir)]
        sys.modules[name] = pkg
        for sub in [k for k in sys.modules if k.startswith(name + ".")]:
            del sys.modules[sub]
    tokenizer_mod = importlib.import_module(f"{name}.tokenizer")
    model_mod = importlib.import_module(f"{name}.model")
    return tokenizer_mod, model_mod


def _split_words(text: str, max_chars: int) -> list[str]:
    """Split on spaces into chunks of at most max_chars (CATT works best on short inputs)."""
    chunks, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class CattDiacritizer:
    """Batch diacritization with CATT's Egyptian (ECA) model."""

    def __init__(
        self,
        catt_dir: str | Path,
        device: str | None = None,
        batch_size: int = 128,
        max_chunk_chars: int = 120,
    ):
        import torch

        self.catt_dir = Path(catt_dir).resolve()
        weights = self.catt_dir / "checkpoints" / "eca_model_weights.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"CATT ECA weights not found at {weights}. Upload the catt_tashkeel "
                "directory including checkpoints/eca_model_weights.pt."
            )
        tokenizer_mod, model_mod = _import_catt_modules(self.catt_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer_mod.TashkeelTokenizer()
        # Same hyper-parameters as catt_tashkeeler.CattTashkeeler.
        self.model = model_mod.TashkeelModel(
            tokenizer=self.tokenizer,
            max_seq_len=1024,
            d_model=512,
            n_layers=6,
            n_heads=16,
            learnable_pos_emb=True,
        )
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_chunk_chars = max_chunk_chars
        logger.info(f"CATT ECA model loaded from {weights} on {self.device}")

    def raw(self, texts: list[str]) -> list[str]:
        """CATT's raw output (special symbols included) for undiacritized texts."""
        import torch

        chunk_of: list[tuple[int, int]] = []  # (text index, number of chunks)
        flat: list[str] = []
        for i, text in enumerate(texts):
            chunks = _split_words(text, self.max_chunk_chars)
            chunk_of.append((i, len(chunks)))
            flat.extend(chunks)
        with torch.no_grad():
            outputs = self.model.do_tashkeel_batch_v2(flat, self.batch_size, verbose=False)
        result, pos = [], 0
        for _, n in chunk_of:
            result.append(" ".join(outputs[pos : pos + n]))
            pos += n
        return result

    def diacritize(self, texts: list[str]) -> list[str | None]:
        """Clean diacritized text per input, or None when CATT changed the letters.

        Inputs must be undiacritized letters and spaces (the pipeline feeds the
        normalized transcript without punctuation). The output has exactly the
        same words, each carrying CATT's diacritics.
        """
        results: list[str | None] = []
        for text, raw in zip(texts, self.raw(texts), strict=True):
            clean = normalize_catt_output(raw)
            if strip_diacritics(clean).split() != text.split():
                results.append(None)
            else:
                results.append(" ".join(clean.split()))
        return results
