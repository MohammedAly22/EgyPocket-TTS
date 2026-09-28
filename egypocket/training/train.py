"""Train EgyPocket-TTS (teacher fine-tune or depth distillation) from a YAML config.

    python -m egypocket.training.train configs/teacher_24l.yaml
    torchrun --nproc-per-node N -m egypocket.training.train configs/teacher_24l.yaml

The loop follows pocket-tts training/train.py (MIT, Kyutai); differences:
  * batches come from the precomputed latent cache (no audio, no Mimi encoding);
  * TensorBoard: losses, lr, grad norm, throughput, GPU memory, validation,
    and audio samples (EMA weights) of fixed monitoring sentences;
  * `touch <run_dir>/STOP` (or SIGTERM/SIGINT) checkpoints and exits cleanly;
  * resuming is automatic from the newest checkpoint in run_dir.
"""

import os

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import contextlib
import logging
import math
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType

import numpy as np
import soundfile
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from egypocket.data.cache import mimi_encode_hash
from egypocket.data.jsonl import read_json
from egypocket.data.loader import Batch, CachedBatches, LatentCache
from egypocket.text.frontend import normalize_text
from egypocket.training.args import TrainArgs, dump_args, load_args, save_args
from egypocket.training.builders import build_models
from egypocket.training.checkpointing import EMA, latest_checkpoint, load_checkpoint, save_checkpoint
from egypocket.training.distributed import (
    avg_across_ranks,
    get_rank,
    get_world_size,
    init_distributed,
    is_torchrun,
    shutdown_distributed,
)
from egypocket.training.model import TrainableTTS

logger = logging.getLogger("train")
LOG_FORMAT = "[%(asctime)s %(levelname)s %(name)s] %(message)s"
VERBOSE_STEPS = 10


def setup_logging(run_dir: Path, rank: int):
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%d-%m %H:%M:%S")
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if rank == 0 else f"_rank{rank}"
    handler = logging.FileHandler(log_dir / f"train_{time.strftime('%Y%m%d-%H%M%S')}{suffix}.log")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%d-%m %H:%M:%S"))
    logging.getLogger().addHandler(handler)


def lr_at(step: int, args: TrainArgs) -> float:
    lr = args.optim.lr
    if step < args.optim.warmup_steps:
        return lr * (step + 1) / args.optim.warmup_steps
    if args.optim.schedule == "cosine":
        progress = (step - args.optim.warmup_steps) / max(1, args.max_steps - args.optim.warmup_steps)
        floor = lr * args.optim.lr_min_ratio
        return floor + 0.5 * (lr - floor) * (1 + math.cos(math.pi * min(1.0, progress)))
    return lr


def compile_model(model: TrainableTTS):
    """Per-layer in-place compilation (pocket-tts): state_dict keys stay unchanged."""
    for layer in model.flow_lm.transformer.layers:
        layer.compile(dynamic=True)
    model.flow_lm.flow_net.compile(dynamic=True)
    if model.distill_teacher is not None:
        for layer in model.distill_teacher.transformer.layers:
            layer.compile(dynamic=True)


@contextlib.contextmanager
def ema_weights(model: nn.Module, ema: EMA | None, enabled: bool) -> Iterator[None]:
    """Temporarily load the EMA shadow into the model (for samples = exported weights)."""
    if ema is None or not enabled:
        yield
        return
    backup = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema.shadow:
                backup[name] = p.detach().clone()
                p.copy_(ema.shadow[name].to(p.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])


class Monitor:
    """TensorBoard writer (rank 0 only)."""

    def __init__(self, run_dir: Path, enabled: bool):
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    def scalars(self, prefix: str, values: dict[str, float], step: int):
        if self.writer is None:
            return
        for k, v in values.items():
            if v is not None and math.isfinite(v):
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def audio(self, tag: str, wav: np.ndarray, step: int, sample_rate: int):
        if self.writer is not None:
            peak = float(np.abs(wav).max()) if wav.size else 0.0
            self.writer.add_audio(tag, wav / max(1.0, peak), step, sample_rate=sample_rate)

    def text(self, tag: str, text: str, step: int):
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()


@torch.no_grad()
def decode_latents(mimi, latents: torch.Tensor) -> np.ndarray:
    """[T, C] raw latents -> mono float32 waveform."""
    from pocket_tts.modules.stateful_module import init_states

    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, 1, (latents.shape[0] + 8) * ratio)
    audio = mimi.decode_from_latent(latents[None].float(), state)[0, 0]
    return audio.float().cpu().numpy()


@torch.no_grad()
def write_samples(
    model: TrainableTTS,
    mimi,
    tokenize,
    args: TrainArgs,
    sentences: list[str],
    voice: torch.Tensor,
    step: int,
    monitor: Monitor,
    ema: EMA | None,
):
    out_dir = args.run_dir / "samples"
    out_dir.mkdir(exist_ok=True)
    model.eval()
    tokens = [torch.tensor(tokenize(s), dtype=torch.long) for s in sentences]
    max_frames = int(args.sample_max_sec * mimi.frame_rate)
    t0 = time.time()
    with ema_weights(model, ema, args.sample_use_ema):
        outs = model.generate(
            tokens,
            [voice] * len(tokens),
            max_frames=max_frames,
            temp=args.sample_temp,
            cfg_coef=args.sample_cfg_coef,
        )
    lengths = []
    for i, latents in enumerate(outs):
        lengths.append(latents.shape[0] / mimi.frame_rate)
        if latents.shape[0] < 8:
            logger.warning(f"sample {i} at step {step}: empty generation, skipped")
            continue
        wav = decode_latents(mimi, latents)
        soundfile.write(str(out_dir / f"step{step:08d}_{i}.wav"), wav, mimi.sample_rate)
        monitor.audio(f"samples/{i}", wav, step, mimi.sample_rate)
    monitor.scalars(
        "samples",
        {
            "mean_duration_sec": float(np.mean(lengths)),
            "hit_max_length": float(np.mean([s >= args.sample_max_sec - 0.1 for s in lengths])),
            "generation_sec": time.time() - t0,
        },
        step,
    )
    model.train()
    logger.info(f"wrote {len(outs)} samples at step {step} ({time.time() - t0:.1f}s)")


@torch.no_grad()
def validate(
    model: TrainableTTS,
    cache: LatentCache,
    tokenize,
    args: TrainArgs,
    device: torch.device,
    autocast,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    loader = CachedBatches(
        cache,
        tokenize,
        args.batch_size,
        seed=1234,
        shuffle=False,
        rank=rank,
        world_size=world_size,
        max_duration_sec=args.data.max_duration_sec,
        max_voice_prompt_sec=args.data.max_voice_prompt_sec,
    )
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        latents, mask, prompt, n_prompt = to_device(batch, device)
        with autocast:
            _, metrics = model(
                latents, mask, batch.text_tokens, prompt, num_voice_prompt_frames=n_prompt
            )
        for k, v in metrics.items():
            if v.numel() == 1:
                totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
        if args.num_valid_batches and n >= args.num_valid_batches:
            break
    model.train()
    return {k: avg_across_ranks(v / max(1, n)) for k, v in totals.items()}


def to_device(batch: Batch, device: torch.device):
    return (
        batch.latents.to(device, non_blocking=True),
        batch.mask.to(device, non_blocking=True),
        batch.prompt_latents.to(device, non_blocking=True),
        batch.num_prompt_frames,  # stays on CPU: the row assembly reads it host-side
    )


def sample_voice(cache: LatentCache, max_voice_prompt_sec: float) -> torch.Tensor:
    """A fixed voice prompt for monitoring: the first valid utterance's prefix."""
    u = 0
    utt = cache.utts[u]
    cut_frame = utt["cuts"][-1][0]  # the longest prompt available within the window
    cap = int(max_voice_prompt_sec * cache.frame_rate)
    lat = cache.utterance_latents(u)[: min(cut_frame, cap)]
    return torch.from_numpy(np.ascontiguousarray(lat))


def main(config_path: str):
    args = load_args(config_path)
    device = init_distributed()
    rank, world_size = get_rank(), get_world_size()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN's bf16 SDPA backward emits NaN at larger batch shapes (pocket-tts note).
    torch.backends.cuda.enable_cudnn_sdp(False)
    if rank == 0:
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        logger.info(f"torch {torch.__version__} | {device} ({gpu}) | world size {world_size}")
        logger.info(f"config {config_path}:\n{dump_args(args).rstrip()}")
        save_args(args, run_dir / "args.yaml")

    train_cache = LatentCache(args.data.train_cache)
    valid_cache = LatentCache(args.data.valid_cache) if args.data.valid_cache else None
    if rank == 0:
        logger.info(
            f"train cache: {len(train_cache)} utterances, {train_cache.meta['hours']} h; "
            f"valid cache: {len(valid_cache) if valid_cache else 0} utterances"
        )

    model, mimi, _config = build_models(args)
    cache_hash = train_cache.meta["mimi_hash"]
    if mimi_encode_hash(mimi) != cache_hash:
        raise SystemExit(
            f"the latent cache {args.data.train_cache} was encoded with a different Mimi than "
            f"{args.model_config}: rebuild the cache with the same model config"
        )
    if args.freeze_head:
        for name, p in model.named_parameters():
            if "flow_net." in name:
                p.requires_grad_(False)
    model.to(device)
    mimi.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"flow_lm + objective: {n_params / 1e6:.1f}M trainable params")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=args.optim.betas,
        eps=args.optim.eps,
        weight_decay=args.optim.weight_decay,
        fused=device.type == "cuda",
    )
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    start_step = 0
    ckpt = latest_checkpoint(run_dir)
    if ckpt is not None:
        start_step = load_checkpoint(ckpt, model, optimizer, ema)

    wrapped: nn.Module = model
    if is_torchrun():
        wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=not args.compile)
    if args.compile and device.type == "cuda":
        compile_model(model)

    tokenizer = model.flow_lm.conditioner.tokenizer
    tokenize = tokenizer.encode
    lexicon = frozenset(read_json(args.data.lexicon)) if args.data.lexicon else None
    sentences = [normalize_text(s, lexicon) for s in args.sample_sentences]
    voice = sample_voice(valid_cache or train_cache, args.data.max_voice_prompt_sec).to(device)

    monitor = Monitor(run_dir, enabled=rank == 0)
    if rank == 0 and start_step == 0:
        monitor.text("config", "```\n" + dump_args(args) + "\n```", 0)
        monitor.text("samples/sentences", "\n\n".join(f"{i}: {s}" for i, s in enumerate(sentences)), 0)
        ref_cache = valid_cache or train_cache
        ref = torch.from_numpy(ref_cache.utterance_latents(0).copy()).to(device)
        monitor.audio("reference/mimi_resynthesis", decode_latents(mimi, ref), 0, mimi.sample_rate)
        monitor.audio("reference/voice_prompt", decode_latents(mimi, voice), 0, mimi.sample_rate)
        monitor.text(
            "reference/text", " ".join(w["word"] for w in ref_cache.utts[0]["words"]), 0
        )

    train_batches = iter(
        CachedBatches(
            train_cache,
            tokenize,
            args.batch_size,
            seed=args.seed + start_step + 1000 * rank,
            shuffle=args.data.shuffle,
            rank=rank,
            world_size=world_size,
            max_duration_sec=args.data.max_duration_sec,
            max_voice_prompt_sec=args.data.max_voice_prompt_sec,
            prompt_trim_max_sec=args.data.prompt_trim_max_sec,
            final_punct_dropout=args.data.final_punct_dropout,
            num_bucket_batches=args.data.num_bucket_batches,
            diacritics=args.data.diacritics,
        )
    )
    use_amp = device.type == "cuda" and args.amp_dtype == "bfloat16"
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    stop_requested = False

    def request_stop(signum: int, frame: FrameType | None):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    stop_file = run_dir / "STOP"
    if rank == 0 and stop_file.exists():
        stop_file.unlink()

    model.train()
    if rank == 0:
        logger.info(f"starting at step {start_step}; the first steps include torch.compile warm-up")
    last_log, steps_since_log, frames_since_log = time.time(), 0, 0
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum_steps):
            batch = next(train_batches)
            latents, mask, prompt, n_prompt = to_device(batch, device)
            frames_since_log += int(batch.mask.sum())
            with autocast:
                loss, metrics = wrapped(
                    latents,
                    mask,
                    batch.text_tokens,
                    prompt,
                    update_stats=step < args.stats_ema_steps,
                    num_voice_prompt_frames=n_prompt,
                )
            scaled = loss / args.grad_accum_steps
            last_micro = micro == args.grad_accum_steps - 1
            if not last_micro and isinstance(wrapped, DDP):
                with wrapped.no_sync():
                    scaled.backward()
            else:
                scaled.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.max_norm)
        if not torch.isfinite(grad_norm):
            raise SystemExit(f"non-finite gradient at step {step}; resume from the last checkpoint")
        optimizer.step()
        if ema is not None:
            ema.update(model)
        steps_since_log += 1
        done = step + 1

        if rank == 0 and stop_file.exists():
            stop_requested = True
        stop = torch.tensor(float(stop_requested), device=device)
        if world_size > 1:
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)

        if rank == 0 and (done - start_step <= VERBOSE_STEPS or done % args.log_freq == 0):
            now = time.time()
            elapsed = max(1e-6, now - last_log)
            values = {k: v.item() for k, v in metrics.items() if v.numel() == 1}
            speed = steps_since_log / elapsed
            logger.info(
                f"step {done} | lr {lr:.2e} | grad {grad_norm:.2f} | {speed:.2f} it/s | "
                + " ".join(f"{k} {v:.4f}" for k, v in values.items())
            )
            monitor.scalars("train", values, done)
            extra = {
                "lr": lr,
                "grad_norm": float(grad_norm),
                "steps_per_sec": speed,
                "audio_sec_per_sec": frames_since_log / train_cache.frame_rate / elapsed,
            }
            if device.type == "cuda":
                extra["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            monitor.scalars("optim", extra, done)
            last_log, steps_since_log, frames_since_log = now, 0, 0

        if stop.item():
            if rank == 0:
                logger.info(f"stop requested: checkpointing step {done} before exit")
                save_checkpoint(run_dir, done, model, optimizer, ema, args.num_ckpt_keep, mimi)
                if stop_file.exists():
                    stop_file.unlink()
            monitor.flush()
            shutdown_distributed()
            return

        if valid_cache is not None and done % args.valid_freq == 0:
            valid_metrics = validate(
                model, valid_cache, tokenize, args, device, autocast, rank, world_size
            )
            if rank == 0:
                logger.info(
                    f"valid @ {done}: " + " ".join(f"{k} {v:.4f}" for k, v in valid_metrics.items())
                )
                monitor.scalars("valid", valid_metrics, done)
        if rank == 0 and args.sample_sentences and done % args.sample_freq == 0:
            write_samples(model, mimi, tokenize, args, sentences, voice, done, monitor, ema)
        if rank == 0 and done % args.ckpt_freq == 0:
            save_checkpoint(run_dir, done, model, optimizer, ema, args.num_ckpt_keep, mimi)
            monitor.flush()

    if rank == 0:
        if start_step < args.max_steps and args.max_steps % args.ckpt_freq != 0:
            save_checkpoint(run_dir, args.max_steps, model, optimizer, ema, args.num_ckpt_keep, mimi)
        if args.sample_sentences and args.max_steps % args.sample_freq != 0:
            write_samples(model, mimi, tokenize, args, sentences, voice, args.max_steps, monitor, ema)
        monitor.flush()
        logger.info("training finished")
    shutdown_distributed()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m egypocket.training.train <config.yaml>")
    main(sys.argv[1])
