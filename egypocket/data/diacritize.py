"""Steps 3-5: CATT diacritization and taa-marbuta restoration.

Pass 1 runs CATT on the corpus as released (ة written ه) and is used only to
vote which word types end in a taa marbuta. After restoring ة, pass 2 runs
CATT again on the restored text — CATT reads restored text better
("ورحمة" -> وَرَحْمَةُ instead of the verb وَرَحِمَهُ) — and its output becomes
the diacritized variant of every word used during training.
"""

import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from egypocket.data.jsonl import read_jsonl, write_json, write_jsonl
from egypocket.paths import SPLITS, DataPaths
from egypocket.text.catt import CattDiacritizer
from egypocket.text.ta_marbuta import build_lexicon, restore

logger = logging.getLogger(__name__)


def _source_manifest(paths: DataPaths, split: str, pass_id: int) -> Path:
    return paths.manifest(split) if pass_id == 1 else paths.restored(split)


def run_catt(
    paths: DataPaths,
    pass_id: int,
    catt_dir: str | Path,
    device: str | None = None,
    batch_size: int = 128,
    block: int = 2048,
) -> dict[str, float]:
    """Diacritize every manifest text; writes work/catt_pass<N>.jsonl ({"id", "diac"})."""
    rows = []
    for split in SPLITS.values():
        src = _source_manifest(paths, split, pass_id)
        if not src.exists():
            raise FileNotFoundError(f"{src} missing: run the previous step first")
        rows.extend(read_jsonl(src))
    catt = CattDiacritizer(catt_dir, device=device, batch_size=batch_size)
    out = []
    failed = 0
    for start in tqdm(range(0, len(rows), block), desc=f"CATT pass {pass_id}"):
        chunk = rows[start : start + block]
        diacs = catt.diacritize([r["text"] for r in chunk])
        for r, d in zip(chunk, diacs, strict=True):
            failed += d is None
            out.append({"id": r["id"], "diac": d})
    write_jsonl(paths.catt(pass_id), out)
    stats = {"texts": len(out), "failed": failed}
    logger.info(f"CATT pass {pass_id}: {stats}")
    return stats


def restore_ta_marbuta(paths: DataPaths, min_ratio: float = 0.6) -> dict[str, object]:
    """Vote per word type on pass-1 output, write restored manifests + the lexicon."""
    diac = {r["id"]: r["diac"] for r in read_jsonl(paths.catt(1))}
    manifests = {split: read_jsonl(paths.manifest(split)) for split in SPLITS.values()}
    all_rows = [r for rows in manifests.values() for r in rows]
    lexicon, votes = build_lexicon(((r["text"], diac.get(r["id"])) for r in all_rows), min_ratio)

    freq: Counter[str] = Counter()
    for r in all_rows:
        freq.update(r["text"].split())
    restored_tokens = sum(freq[w] for w in lexicon)
    for split, rows in manifests.items():
        write_jsonl(
            paths.restored(split), ({**r, "text": restore(r["text"], lexicon)} for r in rows)
        )
    write_json(paths.lexicon, sorted(lexicon))

    def describe(word: str) -> dict[str, object]:
        return {
            "word": word,
            "count": freq[word],
            "votes_ta": votes.positive.get(word, 0),
            "votes_ha": votes.negative.get(word, 0),
            "abstain": votes.abstain.get(word, 0),
        }

    candidates = set(votes.positive) | set(votes.negative) | set(votes.abstain)
    report = {
        "restored_types": len(lexicon),
        "restored_tokens": restored_tokens,
        "candidate_types": len(candidates),
        "top_restored": [describe(w) for w in sorted(lexicon, key=lambda w: -freq[w])[:60]],
        "top_kept": [
            describe(w) for w in sorted(candidates - lexicon, key=lambda w: -freq[w])[:60]
        ],
    }
    write_json(paths.work / "ta_marbuta_report.json", report)
    logger.info(
        f"taa marbuta restored on {len(lexicon)} word types ({restored_tokens} tokens) "
        f"out of {len(candidates)} candidate types"
    )
    return report
