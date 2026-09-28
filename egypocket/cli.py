"""Command line entry points (every notebook cell calls one of these).

    python -m egypocket.cli download
    python -m egypocket.cli manifest
    python -m egypocket.cli catt --pass 1 --catt-dir /workspace/catt_tashkeel
    python -m egypocket.cli restore
    python -m egypocket.cli catt --pass 2 --catt-dir /workspace/catt_tashkeel
    python -m egypocket.cli tokenizer
    python -m egypocket.cli align
    python -m egypocket.cli configs
    python -m egypocket.cli cache
    python -m egypocket.cli prepare --catt-dir ...        # all of the above, skipping finished steps
    python -m egypocket.cli export --run runs/distill_6l --name egypocket_6l --layers 6
    python -m egypocket.cli say --bundle export/egypocket_6l --text "..." --out out.wav

The data root is $EGY_DATA (default /workspace/egypocket_data) or --data-root.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from egypocket.paths import SPLITS, DataPaths

logger = logging.getLogger("egypocket")


def step_download(paths: DataPaths, a: argparse.Namespace):
    from egypocket.data.manifest import download_dataset

    download_dataset(paths, max_workers=a.workers)


def step_manifest(paths: DataPaths, a: argparse.Namespace):
    from egypocket.data.manifest import build_manifests

    stats = build_manifests(paths, min_confidence=a.min_confidence)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def step_catt(paths: DataPaths, a: argparse.Namespace):
    from egypocket.data.diacritize import run_catt

    if not a.catt_dir:
        raise SystemExit("--catt-dir is required (the uploaded catt_tashkeel directory)")
    run_catt(paths, a.pass_id, a.catt_dir, device=a.device, batch_size=a.batch_size)


def step_restore(paths: DataPaths, a: argparse.Namespace):
    from egypocket.data.diacritize import restore_ta_marbuta

    report = restore_ta_marbuta(paths)
    print(
        f"restored {report['restored_types']} word types / {report['restored_tokens']} tokens "
        f"(of {report['candidate_types']} candidate types)"
    )


def step_tokenizer(paths: DataPaths, a: argparse.Namespace):
    from egypocket.text.tokenizer import build_tokenizer_json

    size = build_tokenizer_json(paths.tokenizer)
    print(f"wrote {paths.tokenizer} (vocab {size})")


def step_align(paths: DataPaths, a: argparse.Namespace):
    from egypocket.data.align import align_all

    stats = align_all(paths, model_name=a.model, device=a.device, batch_size=a.batch_size)
    print(json.dumps(stats, indent=2))


def step_configs(paths: DataPaths, a: argparse.Namespace):
    from egypocket.modelcfg import write_model_configs

    for layers, path in write_model_configs(paths).items():
        print(f"{layers} layers: {path}")


def step_cache(paths: DataPaths, a: argparse.Namespace):
    from egypocket.data.cache import build_all_caches

    summary = build_all_caches(
        paths,
        paths.model_config(24),
        drop_worst_fraction=a.drop_worst,
        min_ctc_score=a.min_ctc_score,
        batch_size=a.batch_size,
        device=a.device,
    )
    print(json.dumps(summary, indent=2))


def step_prepare(paths: DataPaths, a: argparse.Namespace):
    """Every data step in order; steps whose outputs exist are skipped (unless --force)."""
    raw_meta = paths.raw / "metadata" / "train.jsonl"
    steps = [
        ("download", lambda: raw_meta.exists() and any((paths.raw / "clips").glob("*/*.wav")), step_download),
        ("manifest", lambda: all(paths.manifest(s).exists() for s in SPLITS.values()), step_manifest),
        ("catt pass 1", lambda: paths.catt(1).exists(), lambda p, x: step_catt(p, _with(x, pass_id=1, batch_size=128))),
        ("restore", lambda: all(paths.restored(s).exists() for s in SPLITS.values()), step_restore),
        ("catt pass 2", lambda: paths.catt(2).exists(), lambda p, x: step_catt(p, _with(x, pass_id=2, batch_size=128))),
        ("tokenizer", lambda: paths.tokenizer.exists(), step_tokenizer),
        ("align", lambda: all(paths.aligned(s).exists() for s in SPLITS.values()), step_align),
        ("configs", lambda: paths.model_config(24).exists() and paths.model_config(6).exists(), step_configs),
        ("cache", lambda: (paths.cache_dir("valid") / "meta.json").exists(), step_cache),
    ]
    for name, done, fn in steps:
        if done() and not a.force:
            print(f"[skip] {name}: outputs exist")
            continue
        print(f"[run ] {name}")
        fn(paths, a)


def _with(ns: argparse.Namespace, **kw) -> argparse.Namespace:
    d = vars(ns).copy()
    d.update(kw)
    return argparse.Namespace(**d)


def step_export(paths: DataPaths, a: argparse.Namespace):
    from egypocket.bundle import export_bundle

    out = export_bundle(paths, a.run, a.name, a.layers, checkpoint=a.checkpoint, voice_wav=a.voice)
    print(f"bundle written to {out}")


def step_say(paths: DataPaths, a: argparse.Namespace):
    from egypocket.inference import EgyPocketTTS

    tts = EgyPocketTTS(a.bundle, num_threads=a.threads, quantize=a.quantize)
    tts.save(a.text, a.out, voice=a.voice)
    s = tts.last_stats
    print(
        f"wrote {a.out}: {s.audio_sec:.2f}s audio in {s.wall_sec:.2f}s "
        f"(RTF x{s.real_time_factor:.1f}, first audio after {s.first_chunk_sec * 1000:.0f} ms)"
    )


def main(argv: list[str] | None = None):
    # Batches grow in length (clips are length-sorted); expandable segments stop the
    # CUDA caching allocator from fragmenting. Must be set before torch initializes CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")
    p = argparse.ArgumentParser(prog="egypocket")
    p.add_argument("--data-root", default=None, help="defaults to $EGY_DATA or /workspace/egypocket_data")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download")
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(fn=step_download)

    s = sub.add_parser("manifest")
    s.add_argument("--min-confidence", type=float, default=0.93)
    s.set_defaults(fn=step_manifest)

    s = sub.add_parser("catt")
    s.add_argument("--pass", dest="pass_id", type=int, choices=(1, 2), required=True)
    s.add_argument("--catt-dir", required=True)
    s.add_argument("--device", default=None)
    s.add_argument("--batch-size", type=int, default=128)
    s.set_defaults(fn=step_catt)

    s = sub.add_parser("restore")
    s.set_defaults(fn=step_restore)

    s = sub.add_parser("tokenizer")
    s.set_defaults(fn=step_tokenizer)

    s = sub.add_parser("align")
    s.add_argument("--model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    s.add_argument("--device", default=None)
    s.add_argument("--batch-size", type=int, default=16)
    s.set_defaults(fn=step_align)

    s = sub.add_parser("configs")
    s.set_defaults(fn=step_configs)

    s = sub.add_parser("cache")
    s.add_argument("--drop-worst", type=float, default=0.03, help="fraction of clips with the lowest CTC score to drop")
    s.add_argument("--min-ctc-score", type=float, default=None, help="absolute cut-off instead of --drop-worst")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=step_cache)

    s = sub.add_parser("prepare")
    s.add_argument("--catt-dir", required=True)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--min-confidence", type=float, default=0.93)
    s.add_argument("--model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    s.add_argument("--drop-worst", type=float, default=0.03)
    s.add_argument("--min-ctc-score", type=float, default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=step_prepare, pass_id=1)

    s = sub.add_parser("export")
    s.add_argument("--run", required=True, help="training run directory")
    s.add_argument("--name", required=True)
    s.add_argument("--layers", type=int, default=6)
    s.add_argument("--checkpoint", default=None, help="default: newest checkpoint of the run")
    s.add_argument("--voice", default=None, help="wav for the default voice (default: a held-out clip)")
    s.set_defaults(fn=step_export)

    s = sub.add_parser("say")
    s.add_argument("--bundle", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--out", default="out.wav")
    s.add_argument("--voice", default=None)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--quantize", action="store_true")
    s.set_defaults(fn=step_say)

    a = p.parse_args(argv)
    if a.data_root:
        os.environ["EGY_DATA"] = str(Path(a.data_root).resolve())
    paths = DataPaths.from_env()
    paths.mkdirs()
    a.fn(paths, a)


if __name__ == "__main__":
    sys.exit(main())
