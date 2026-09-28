"""Step 1-2: download the corpus and build clean, normalized manifests.

Manifest rows (work/manifest_<split>.jsonl):
    {"id", "audio" (absolute wav path), "duration", "split", "text", "raw_text", "confidence"}
`text` is the frontend-normalized transcript WITHOUT punctuation (pause
punctuation is added later from the audio), numbers verbalized the Egyptian way.
"""

import logging
from collections import Counter
from pathlib import Path

from egypocket.data.jsonl import read_jsonl, write_json, write_jsonl
from egypocket.paths import DATASET_REPO, SPLITS, DataPaths
from egypocket.text.chars import LETTERS_SET
from egypocket.text.frontend import normalize_text_verbose, strip_diacritics

logger = logging.getLogger(__name__)


def download_dataset(paths: DataPaths, repo_id: str = DATASET_REPO, max_workers: int = 16) -> Path:
    """Snapshot the wav clips + metadata (~12.4 GB). Resumable: finished files are skipped."""
    from huggingface_hub import snapshot_download

    paths.raw.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(paths.raw),
        allow_patterns=["metadata/*.jsonl", "metadata/*.json", "clips/**/*.wav", "README.md"],
        max_workers=max_workers,
    )
    n_wav = sum(1 for _ in Path(local, "clips").rglob("*.wav"))
    logger.info(f"dataset at {local}: {n_wav} wav files")
    return Path(local)


def plain_model_text(raw_text: str) -> tuple[str, list[str]]:
    """Normalized transcript with punctuation removed (letters and single spaces)."""
    text, dropped = normalize_text_verbose(raw_text)
    text = strip_diacritics(text).replace(",", " ").replace(".", " ")
    return " ".join(text.split()), dropped


def build_manifests(
    paths: DataPaths,
    min_confidence: float = 0.93,
    min_duration: float = 1.5,
    max_duration: float = 20.0,
    min_words: int = 3,
    min_letters_per_sec: float = 4.0,
    max_letters_per_sec: float = 20.0,
    drop_promo: bool = True,
) -> dict[str, dict[str, float]]:
    """Filter + normalize every official split. Returns per-split statistics."""
    stats: dict[str, dict[str, float]] = {}
    for src_split, split in SPLITS.items():
        meta = paths.raw / "metadata" / f"{src_split}.jsonl"
        if not meta.exists():
            raise FileNotFoundError(f"{meta} missing: run the download step first")
        reasons: Counter[str] = Counter()
        kept = []
        for row in read_jsonl(meta):
            audio = paths.raw / row["audio_path"]
            duration = float(row["duration"])
            if drop_promo and row.get("promo"):
                reasons["promo"] += 1
                continue
            if float(row.get("confidence", 1.0)) < min_confidence:
                reasons["low_confidence"] += 1
                continue
            if not min_duration <= duration <= max_duration:
                reasons["duration"] += 1
                continue
            text, dropped = plain_model_text(row["text"])
            if dropped:
                reasons["unsupported_characters"] += 1
                continue
            if len(text.split()) < min_words:
                reasons["too_few_words"] += 1
                continue
            letters = sum(c in LETTERS_SET for c in text)
            if not min_letters_per_sec <= letters / duration <= max_letters_per_sec:
                reasons["implausible_rate"] += 1
                continue
            if not audio.exists():
                reasons["missing_audio"] += 1
                continue
            kept.append(
                {
                    "id": row["id"],
                    "audio": str(audio.resolve()),
                    "duration": duration,
                    "split": split,
                    "text": text,
                    "raw_text": row["text"],
                    "confidence": float(row.get("confidence", 1.0)),
                }
            )
        write_jsonl(paths.manifest(split), kept)
        hours = sum(r["duration"] for r in kept) / 3600
        stats[split] = {"kept": len(kept), "hours": round(hours, 2), **dict(reasons)}
        logger.info(f"{split}: kept {len(kept)} clips ({hours:.2f} h), dropped {dict(reasons)}")
    write_json(paths.work / "manifest_stats.json", stats)
    return stats
