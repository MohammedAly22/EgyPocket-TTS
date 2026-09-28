# Adapted from kyutai-labs/pocket-tts (training/), MIT licence, Copyright (c) Kyutai.
# See NOTICE.md. Local changes are limited to import paths unless stated below.
"""Build a shallower checkpoint from a deeper one by selecting layers.

Every tensor outside the backbone is copied unchanged; the backbone keeps the
teacher's bottom and top layers (see select_layers). Use it to initialize a
depth-distillation student -- starting from the teacher's own layers converges
far faster than a random backbone.
"""

import re

import torch


def select_layers(n_teacher: int, n_student: int) -> list[int]:
    """Which teacher layers the student starts from: the bottom half and the
    top half of the stack ("ends").

    Won the 4-way ablation (ends/last/spaced/first, 2026-08): top-heavy seeds
    reach teacher parity ~20k steps sooner than bottom-heavy ones, and ends
    edged out last at every checkpoint from 30k on.
    """
    assert 0 < n_student <= n_teacher, f"cannot shrink {n_teacher} -> {n_student}"
    head = (n_student + 1) // 2
    tail = n_student - head
    return list(range(head)) + list(range(n_teacher - tail, n_teacher))


def shrink(
    state: dict[str, torch.Tensor], n_student: int
) -> tuple[dict[str, torch.Tensor], list[int]]:
    pattern = re.compile(r"^(flow_lm\.)?transformer\.layers\.(\d+)\.(.*)$")
    depth = max((int(m.group(2)) for m in (pattern.match(k) for k in state) if m), default=-1) + 1
    assert depth > 0, "no transformer layers found in the checkpoint"
    keep = select_layers(depth, n_student)
    remap = {old: new for new, old in enumerate(keep)}
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        m = pattern.match(k)
        if not m:
            out[k] = v
            continue
        old = int(m.group(2))
        if old not in remap:
            continue
        out[f"{m.group(1) or ''}transformer.layers.{remap[old]}.{m.group(3)}"] = v
    return out, keep


