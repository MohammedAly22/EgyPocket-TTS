"""Helpers for the Jupyter notebooks: launch/monitor/stop training, TensorBoard, samples.

Training runs as a detached background process (it survives a closed browser
tab or a restarted kernel); the notebooks only start it, watch it and stop it.
"""

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pid_file(run_dir: Path) -> Path:
    return Path(run_dir) / "train.pid"


def is_running(run_dir: str | Path) -> bool:
    pid_file = _pid_file(Path(run_dir))
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A recycled pid would not be ours: check the command line on Linux.
    cmdline = Path(f"/proc/{pid}/cmdline")
    return not cmdline.exists() or b"egypocket.training.train" in cmdline.read_bytes()


def launch_training(config: str | Path, run_dir: str | Path, extra_env: dict[str, str] | None = None) -> int:
    """Start `python -m egypocket.training.train <config>` detached; returns its pid."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_running(run_dir):
        raise RuntimeError(f"a training process is already running for {run_dir}")
    env = {**os.environ, **(extra_env or {})}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    log = open(run_dir / "stdout.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "egypocket.training.train", str(config)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _pid_file(run_dir).write_text(str(proc.pid))
    return proc.pid


def stop_training(run_dir: str | Path, hard: bool = False):
    """Graceful: the trainer checkpoints the current step and exits. hard=True kills it."""
    run_dir = Path(run_dir)
    if hard and is_running(run_dir):
        os.killpg(int(_pid_file(run_dir).read_text()), signal.SIGKILL)
        return
    if not is_running(run_dir):
        print(f"no training process is running for {run_dir}")
        return
    (run_dir / "STOP").touch()
    print("stop requested: the trainer checkpoints the current step and exits")


def tail(path: str | Path, n: int = 30) -> str:
    path = Path(path)
    if not path.exists():
        return f"({path} does not exist yet)"
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def status(run_dir: str | Path, n: int = 25) -> str:
    run_dir = Path(run_dir)
    state = "RUNNING" if is_running(run_dir) else "not running"
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
    last = ckpts[-1].name if ckpts else "none"
    return f"[{state}] newest checkpoint: {last}\n" + tail(run_dir / "stdout.log", n)


def start_tensorboard(logdir: str | Path, port: int = 6006) -> str:
    """Start TensorBoard in the background (idempotent) and return how to open it."""
    logdir = Path(logdir)
    running = subprocess.run(["pgrep", "-f", f"tensorboard.*--port {port}"], capture_output=True)
    if running.returncode != 0:
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port),
             "--bind_all", "--reload_interval", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod:
        return (
            f"TensorBoard: https://{pod}-{port}.proxy.runpod.net  "
            f"(expose HTTP port {port} in the pod settings if the link does not open)"
        )
    return f"TensorBoard: http://localhost:{port}"


def read_scalars(run_dir: str | Path) -> dict[str, tuple[list[int], list[float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb = Path(run_dir) / "tb"
    out: dict[str, tuple[list[int], list[float]]] = {}
    for events in sorted(tb.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            steps, values = out.setdefault(tag, ([], []))
            for e in acc.Scalars(tag):
                steps.append(e.step)
                values.append(e.value)
    for tag, (steps, values) in out.items():
        order = sorted(range(len(steps)), key=steps.__getitem__)
        out[tag] = ([steps[i] for i in order], [values[i] for i in order])
    return out


def plot_scalars(run_dir: str | Path, patterns: list[str] | None = None, smooth: int = 10):
    import matplotlib.pyplot as plt
    import numpy as np

    scalars = read_scalars(run_dir)
    patterns = patterns or ["train/loss", "train/flow_loss", "train/eos_loss", "train/distill_mse",
                            "valid/loss", "valid/flow_loss", "valid/distill_mse", "optim/lr",
                            "optim/grad_norm", "optim/steps_per_sec", "samples/mean_duration_sec"]
    tags = [t for t in patterns if t in scalars]
    if not tags:
        print("no scalars yet")
        return
    cols = 3
    rows = (len(tags) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        steps, values = scalars[tag]
        ax.plot(steps, values, alpha=0.35)
        if len(values) > smooth > 1 and tag.startswith("train/"):
            kernel = np.ones(smooth) / smooth
            ax.plot(steps[smooth - 1 :], np.convolve(values, kernel, mode="valid"))
        ax.set_title(tag)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(tags) :]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def latest_samples(run_dir: str | Path) -> tuple[int, list[Path]]:
    files = list((Path(run_dir) / "samples").glob("step*_*.wav"))
    if not files:
        return 0, []
    step = max(int(re.match(r"step(\d+)_", f.name).group(1)) for f in files)
    chosen = sorted(f for f in files if f.name.startswith(f"step{step:08d}_"))
    return step, chosen


def show_samples(run_dir: str | Path, sentences: list[str] | None = None, step: int | None = None):
    """Audio players for one sample step (default: newest)."""
    from IPython.display import Audio, Markdown, display

    run_dir = Path(run_dir)
    if step is None:
        step, files = latest_samples(run_dir)
    else:
        files = sorted((run_dir / "samples").glob(f"step{step:08d}_*.wav"))
    if not files:
        print("no samples yet (they are written every sample_freq steps)")
        return
    display(Markdown(f"**samples at step {step}**"))
    for f in files:
        i = int(f.stem.split("_")[-1])
        label = sentences[i] if sentences and i < len(sentences) else f.name
        display(Markdown(f"`{i}` {label}"))
        display(Audio(str(f)))
