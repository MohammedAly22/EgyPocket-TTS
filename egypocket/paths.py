"""On-disk layout of one EgyPocket data root (default /workspace/egypocket_data).

    raw/                         HF snapshot of the dataset (wav + metadata)
    work/                        intermediate manifests (jsonl), CATT outputs, reports
    artifacts/                   tokenizer.json, ta-marbuta lexicon, pocket model configs
    cache/<split>/               precomputed Mimi latents + training index (what the trainer reads)
    runs/<name>/                 checkpoints, TensorBoard logs, audio samples
    export/<name>/               self-contained CPU inference bundles
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = "/workspace/egypocket_data"
DATASET_REPO = "ehabnegm/100-hour-Egyptian-dataset-single-speaker"
# Dataset split name -> our split name.
SPLITS = {"train": "train", "dev": "valid", "test": "test"}


def data_root() -> Path:
    return Path(os.environ.get("EGY_DATA", DEFAULT_DATA_ROOT))


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def from_env(cls) -> "DataPaths":
        return cls(data_root())

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def export(self) -> Path:
        return self.root / "export"

    def manifest(self, split: str) -> Path:
        return self.work / f"manifest_{split}.jsonl"

    def catt(self, pass_id: int) -> Path:
        return self.work / f"catt_pass{pass_id}.jsonl"

    def restored(self, split: str) -> Path:
        return self.work / f"restored_{split}.jsonl"

    def aligned(self, split: str) -> Path:
        return self.work / f"aligned_{split}.jsonl"

    def cache_dir(self, split: str) -> Path:
        return self.cache / split

    @property
    def tokenizer(self) -> Path:
        return self.artifacts / "tokenizer.json"

    @property
    def lexicon(self) -> Path:
        return self.artifacts / "ta_marbuta_lexicon.json"

    def model_config(self, layers: int) -> Path:
        return self.artifacts / f"egy_{layers}l.yaml"

    def mkdirs(self):
        for p in (self.raw, self.work, self.artifacts, self.cache, self.runs, self.export):
            p.mkdir(parents=True, exist_ok=True)
