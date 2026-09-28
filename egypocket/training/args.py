"""Training configuration (YAML -> dataclasses).

Field names and semantics follow pocket-tts training/args.py (MIT, Kyutai) so the
vendored model/builder code works unchanged. Differences:

* `data` points at precomputed latent caches (see egypocket/data/cache.py)
  instead of audio manifests: training never touches audio.
* `data.diacritics` controls how often training text is diacritized.
* String values may use environment variables, e.g. `${EGY_DATA}/cache/train`.
"""

import dataclasses
import os
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

T = tp.TypeVar("T")


@dataclass
class DiacriticArgs:
    """How training text mixes plain and CATT-diacritized words.

    Per sample one mode is drawn: plain (no diacritics at all), partial (each
    word diacritized with probability drawn from `partial_word_prob`) or full
    (every word). A diacritized word keeps only a random subset of its marks
    with probability `sparse_word_prob`. Defaults put ~15-20% of words in
    diacritized form: the model stays a plain-text reader and diacritics act as
    an override.
    """

    p_plain: float = 0.55
    p_partial: float = 0.30
    p_full: float = 0.15
    partial_word_prob: tuple[float, float] = (0.1, 0.5)
    sparse_word_prob: float = 0.4
    sparse_keep_prob: tuple[float, float] = (0.3, 0.8)

    def __post_init__(self):
        total = self.p_plain + self.p_partial + self.p_full
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"diacritics p_plain + p_partial + p_full must be 1, got {total}")


@dataclass
class DataArgs:
    train_cache: str = ""
    valid_cache: str = ""
    # For the voice-prompt cut windows the cache was built with (checked at start).
    max_voice_prompt_sec: float = 5.0
    max_duration_sec: float = 20.0
    # Robustness augmentations of the released pocket-tts models.
    prompt_trim_max_sec: float = 0.5
    final_punct_dropout: float = 0.3
    num_bucket_batches: int = 20
    shuffle: bool = True
    # ta-marbuta lexicon (applied to the monitoring sentences like at inference).
    lexicon: str = ""
    diacritics: DiacriticArgs = field(default_factory=DiacriticArgs)


@dataclass
class FlowArgs:
    type: str = "lsd"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimArgs:
    lr: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_norm: float = 1.0
    warmup_steps: int = 500
    schedule: str = "constant"  # "constant" | "cosine"
    lr_min_ratio: float = 0.0


DEFAULT_SAMPLE_SENTENCES = [
    "النهارده الجو حلو قوي, وانا رايح الشغل بدري علشان الحق الاجتماع.",
    "ولو ركزنا شوية في الحكاية دي, هنلاقي ان المصريين القدماء كانوا سابقين عصرهم في حاجات كتير.",
    "رحت المدينة امبارح.",
    "رحت المدينةْ امبارح.",
    "العِلْم نور, والعَلَم رمز البلد.",
]


@dataclass
class TrainArgs:
    data: DataArgs = field(default_factory=DataArgs)
    flow: FlowArgs = field(default_factory=FlowArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)

    model_config: str = ""
    model_overrides: dict[str, tp.Any] = field(default_factory=dict)
    start_from_pretrained: bool = True
    freeze_head: bool = False
    reset_text_embedding: bool = False

    run_dir: Path = Path("runs/debug")
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 100_000
    seed: int = 42
    # "bfloat16" on Ampere/Ada/Hopper GPUs; "float32" disables autocast.
    amp_dtype: str = "bfloat16"

    flow_batch_multiplier: int = 1
    eos_loss_weight: float = 0.1
    text_dropout: float = 0.2
    voice_dropout: float = 0.2
    stats_ema_steps: int = 0
    stats_ema_decay: float = 0.999

    log_freq: int = 25
    valid_freq: int = 1000
    num_valid_batches: int = 0  # 0 = the whole validation cache
    compile: bool = True
    sample_sentences: list[str] = field(default_factory=lambda: list(DEFAULT_SAMPLE_SENTENCES))
    sample_freq: int = 1000
    sample_temp: float = 0.3
    sample_cfg_coef: float = 1.0
    sample_max_sec: float = 20.0
    sample_use_ema: bool = True
    ckpt_freq: int = 2000
    num_ckpt_keep: int = 3
    ema_decay: float = 0.999

    distill_cfg_coef: float = 0.0
    distill_teacher_config: str = ""
    distill_teacher_overrides: dict[str, tp.Any] = field(default_factory=dict)
    distill_teacher_weights: str = ""
    distill_teacher_use_ema: bool = True
    student_init_from: str = ""

    def __post_init__(self):
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.num_ckpt_keep < 1:
            raise ValueError(f"num_ckpt_keep must be >= 1, got {self.num_ckpt_keep}")
        for name in ("valid_freq", "ckpt_freq", "log_freq", "sample_freq"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.amp_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"amp_dtype must be bfloat16 or float32, got {self.amp_dtype}")
        if self.distill_cfg_coef > 0 and not (
            self.start_from_pretrained or self.distill_teacher_config
        ):
            raise ValueError(
                "distillation needs a teacher: set distill_teacher_config, or "
                "start_from_pretrained: true to distil a frozen copy of the model itself"
            )
        if self.distill_teacher_config and not self.distill_teacher_weights:
            raise ValueError("distill_teacher_config is set but distill_teacher_weights is not")


_SUB = {"data": DataArgs, "flow": FlowArgs, "optim": OptimArgs, "diacritics": DiacriticArgs}


def _expand(value: str) -> str:
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(f"unresolved environment variable in config value {value!r}")
    return expanded


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    for key, value in (data or {}).items():
        if key not in fields:
            raise ValueError(f"Unknown config key {key!r} for {cls.__name__}")
        ftype = fields[key].type
        if key in _SUB:
            value = _from_dict(_SUB[key], value)
        elif isinstance(value, str):
            value = _expand(value)
        if key in ("betas", "partial_word_prob", "sparse_keep_prob"):
            value = tuple(float(v) for v in value)
        elif ftype is float or ftype == "float":
            value = float(value)
        elif ftype is int or ftype == "int":
            value = int(value)
        elif ftype is Path or ftype == "Path":
            value = Path(value)
        elif key == "sample_sentences":
            value = [str(v) for v in value]
        kwargs[key] = value
    return cls(**kwargs)


def load_args(path: str | Path) -> TrainArgs:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _from_dict(TrainArgs, raw)


def dump_args(args: TrainArgs) -> str:
    def plain(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    return yaml.safe_dump(plain(dataclasses.asdict(args)), sort_keys=False, allow_unicode=True)


def save_args(args: TrainArgs, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(dump_args(args))
