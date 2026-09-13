"""Rig catalog: the hardware side of capacity planning.

Approximate datasheet numbers (dense FP16/BF16 tensor TFLOPS, memory
bandwidth, whole-system draw under sustained load). Documented per-rig in
``docs/model.md``. Extend at runtime with :func:`register_rig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Rig:
    name: str
    label: str
    vram_gb: float
    mem_bw_gbps: float    # HBM/GDDR bandwidth
    tflops_fp16: float    # dense FP16/BF16 tensor throughput
    base_watts: float     # whole-system sustained draw (system, not TDP)
    cpu_cores: int = 8


RIGS: Dict[str, Rig] = {
    "rtx-3060": Rig("rtx-3060", "RTX 3060 12 GB",        12, 337.0,  13.0, 240),
    "rtx-4060": Rig("rtx-4060", "RTX 4060 8 GB",          8, 272.0,  15.1, 165),
    "rtx-3090": Rig("rtx-3090", "RTX 3090 24 GB",        24, 936.0, 142.0, 350),
    "rtx-4080": Rig("rtx-4080", "RTX 4080 16 GB",        16, 717.0, 229.0, 320),
    "rtx-4090": Rig("rtx-4090", "RTX 4090 24 GB",        24, 1008.0, 330.0, 450),
    "rtx-5090": Rig("rtx-5090", "RTX 5090 32 GB",        32, 1792.0, 400.0, 575),
    "a100-40":  Rig("a100-40",  "A100 40 GB PCIe",       40, 1555.0, 312.0, 400),
    "m3-max":   Rig("m3-max",   "Apple M3 Max 36 GB",    36, 400.0,  28.0, 120),
}


def get_rig(name: str) -> Rig:
    if name not in RIGS:
        raise KeyError("unknown rig %r; known: %s" % (name, sorted(RIGS)))
    return RIGS[name]


def register_rig(rig: Rig) -> None:
    RIGS[rig.name] = rig


def all_rigs() -> List[Rig]:
    return sorted(RIGS.values(), key=lambda r: (r.vram_gb, r.name))
