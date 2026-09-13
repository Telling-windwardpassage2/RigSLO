"""SLO planning: from rig + model to latency percentiles and capacity.

Deterministic by construction (no sampling): percentiles are analytic
multipliers on the expected request time, and queueing pressure is a
documented linear penalty once concurrency exceeds the KV-cache capacity.

    p90 = p50 * 1.15        p99 = p50 * 1.35
    queue_factor = 1 + max(0, c - max_sessions) / max_sessions

All multipliers are module constants so a reviewer can challenge exactly
one number instead of a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .models import Model, weights_bytes, kvcache_bytes, vram_bytes, BYTES_PER_QUANT
from .rigs import Rig
from .throughput import decode_toks, prefill_toks

P90_MULT = 1.15
P99_MULT = 1.35


@dataclass(frozen=True)
class Plan:
    model: str
    rig: str
    quant: str
    seq_len: int
    out_tokens: int
    concurrency: int
    vram_gb: float
    fits: bool
    decode_toks: float
    prefill_toks: float
    ttft_ms: float
    gen_ms: float
    total_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_sessions: int
    queue_factor: float
    requests_per_hour: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d


def plan(m: Model, rig: Rig, *, quant: Optional[str] = None, seq_len: int = 8192,
         out_tokens: int = 256, concurrency: int = 1, kv_quant: str = "fp16",
         calibration: float = 1.0, overhead_gb: float = 0.5) -> Plan:
    quant = quant or m.default_quant
    if seq_len <= 0 or out_tokens <= 0 or concurrency <= 0:
        raise ValueError("seq_len, out_tokens, concurrency must be positive")

    vram = vram_bytes(m, seq_len, quant, kv_quant)
    vram_gb = vram / 1e9
    fits_ = vram + int(overhead_gb * 1e9) <= int(rig.vram_gb * 1e9)

    dec = decode_toks(m, rig, quant=quant, seq_len=seq_len, kv_quant=kv_quant, calibration=calibration)
    pre = prefill_toks(m, rig, quant=quant, calibration=calibration)

    ttft_ms = seq_len / pre * 1000.0
    gen_ms = out_tokens / dec * 1000.0
    p50 = ttft_ms + gen_ms
    p90 = p50 * P90_MULT
    p99 = p50 * P99_MULT

    # KV-cache capacity for concurrent sequences.
    w = weights_bytes(m, quant) + int(overhead_gb * 1e9)
    kv_budget = int(rig.vram_gb * 1e9) - w
    per_seq = kvcache_bytes(m, seq_len, kv_quant)
    max_sessions = max(0, kv_budget // per_seq) if per_seq > 0 and kv_budget > 0 else 0

    if max_sessions == 0:
        queue_factor = float("inf") if concurrency > 1 else 1.0
    else:
        excess = max(0, concurrency - max_sessions)
        queue_factor = 1.0 + excess / float(max_sessions)
    if queue_factor == float("inf"):
        p90 = float("inf")
        p99 = float("inf")
        rph = 0.0
    else:
        p90 *= 1.0 + 0.5 * max(0.0, (queue_factor - 1.0))
        p99 *= queue_factor
        rph = 3600.0 * 1000.0 / p50 if p50 > 0 else 0.0

    return Plan(
        model=m.name, rig=rig.name, quant=quant, seq_len=seq_len, out_tokens=out_tokens,
        concurrency=concurrency, vram_gb=vram_gb, fits=fits_,
        decode_toks=dec, prefill_toks=pre,
        ttft_ms=ttft_ms, gen_ms=gen_ms, total_ms=p50,
        p50_ms=p50, p90_ms=p90, p99_ms=p99,
        max_sessions=max_sessions, queue_factor=queue_factor,
        requests_per_hour=rph,
    )


def max_seq_len(m: Model, rig: Rig, quant: str = "int4", kv_quant: str = "fp16",
                overhead_gb: float = 0.5) -> int:
    """Largest single-sequence context that fits (binary search)."""
    w = weights_bytes(m, quant) + int(overhead_gb * 1e9)
    if w >= int(rig.vram_gb * 1e9):
        return 0
    per_token = int(2 * m.n_layers * m.n_kv_heads * m.head_dim * BYTES_PER_QUANT.get(kv_quant, 2.0))
    if per_token <= 0:
        return 0
    return (int(rig.vram_gb * 1e9) - w) // per_token
