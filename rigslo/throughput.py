"""Analytical throughput model.

Two regimes, both roofline-style (same spirit as KernelGym):

* **Decode** is memory-bandwidth-bound: each generated token reads the
  whole weight set once (plus the KV cache for the active context).
      tok/s  ~=  mem_bw / (weights + kvcache(seq)) * DECODE_EFF
* **Prefill** is compute-bound: a forward pass over N tokens costs ~2N
  FLOPs per parameter (one multiply-add per weight).
      tok/s  ~=  tflops / (2 * active_params) * PREFILL_EFF

Efficiency factors absorb kernel quality, Python/engine overhead, and
quantization dequant cost. A probe measurement multiplies both via
``calibration`` so the analytic model anchors to reality.
"""
from __future__ import annotations

from .models import Model, weights_bytes, kvcache_bytes
from .rigs import Rig

DECODE_EFF = 0.85
PREFILL_EFF = 0.40


def decode_toks(m: Model, rig: Rig, quant: str = "int4", seq_len: int = 2048,
                kv_quant: str = "fp16", calibration: float = 1.0) -> float:
    if calibration <= 0:
        raise ValueError("calibration must be positive")
    step_bytes = weights_bytes(m, quant) + kvcache_bytes(m, seq_len, kv_quant)
    if step_bytes <= 0:
        raise ValueError("degenerate model size")
    return rig.mem_bw_gbps * 1e9 / step_bytes * DECODE_EFF * calibration


def prefill_toks(m: Model, rig: Rig, quant: str = "int4", calibration: float = 1.0) -> float:
    if calibration <= 0:
        raise ValueError("calibration must be positive")
    flops_per_token = 2.0 * m.weights_base_bytes
    return rig.tflops_fp16 * 1e12 / flops_per_token * PREFILL_EFF * calibration


def decode_toks_at(m: Model, rig: Rig, seq_len: int) -> float:
    """Alias keeping the signature discoverable for seq-dependent reads."""
    return decode_toks(m, rig, seq_len=seq_len)
