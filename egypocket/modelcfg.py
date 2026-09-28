"""pocket-tts model configs for EgyPocket (architecture + tokenizer + base weights).

* egy_24l.yaml — the teacher: pocket-tts English 24-layer teacher weights
  (english_2026-04_24l), our character tokenizer (the text embedding is
  re-initialized, see configs/teacher_24l.yaml).
* egy_6l.yaml  — the CPU student: same model with 6 layers (~100M params,
  the size of the released Pocket TTS). Its weights come from distillation;
  weights_path only supplies the Mimi codec, identical to the teacher's so the
  latent cache is valid for both.
"""

import copy
from pathlib import Path
from typing import Any

import yaml

from egypocket.paths import DataPaths
from egypocket.text.tokenizer import vocab_size

BASE_TEACHER = "english_2026-04_24l.yaml"


def _base_config() -> dict[str, Any]:
    from pocket_tts.utils.config import CONFIGS_DIR

    return yaml.safe_load((CONFIGS_DIR / BASE_TEACHER).read_text(encoding="utf-8"))


def egy_config(tokenizer_path: str | Path, n_bins: int, num_layers: int) -> dict[str, Any]:
    cfg = copy.deepcopy(_base_config())
    cfg.pop("weights_path_without_voice_cloning", None)  # training needs the real Mimi encoder
    cfg["default_temperature"] = 0.3
    # Arabic has no letter case; a capitalized first letter would be an unknown token.
    cfg["capitalize_first_letter"] = False
    cfg["append_terminal_punctuation"] = True
    cfg["remove_semicolons"] = False
    cfg["pad_with_spaces_for_short_inputs"] = False
    flow_lm = cfg["flow_lm"]
    flow_lm["transformer"]["num_layers"] = num_layers
    flow_lm["lookup_table"] = {
        "dim": flow_lm["lookup_table"]["dim"],
        "n_bins": n_bins,
        "tokenizer": "tokenizers",
        "tokenizer_path": str(tokenizer_path),
    }
    return cfg


def write_model_configs(paths: DataPaths) -> dict[int, Path]:
    from pocket_tts.utils.config import load_config

    if not paths.tokenizer.exists():
        raise FileNotFoundError(f"{paths.tokenizer} missing: run the tokenizer step first")
    n_bins = vocab_size(paths.tokenizer)
    out = {}
    for layers in (24, 6):
        cfg = egy_config(paths.tokenizer.resolve(), n_bins, layers)
        path = paths.model_config(layers)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        load_config(path)  # schema check
        out[layers] = path
    return out


def bundle_config(model_config: str | Path) -> dict[str, Any]:
    """Config for an inference bundle: paths relative to the bundle directory."""
    cfg = yaml.safe_load(Path(model_config).read_text(encoding="utf-8"))
    cfg["weights_path"] = "model.safetensors"
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = "tokenizer.json"
    cfg["flow_lm"].pop("weights_path", None)
    cfg["mimi"].pop("weights_path", None)
    return cfg
