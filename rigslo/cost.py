"""Power and $ arithmetic.

Reference cloud price list is the *comparator*, not a vendor pitch: it lets
"local cost" be read against what the same tokens would cost on a hosted
small model. Update ``CLOUD_REFERENCE`` when list prices move.
"""
from __future__ import annotations

# USD per 1k tokens, reference small hosted model (gpt-4o-mini-class, 2026 list).
CLOUD_REFERENCE: dict = {
    "name": "gpt-4o-mini (reference class)",
    "in_usd_per_1k": 0.15,
    "out_usd_per_1k": 0.60,
}

DEFAULT_USD_PER_KWH = 0.15


def cost_per_request(watts: float, total_seconds: float, usd_per_kwh: float = DEFAULT_USD_PER_KWH) -> float:
    """Energy cost of one request: P * t * price."""
    if watts < 0 or total_seconds < 0:
        raise ValueError("watts and time must be non-negative")
    return (watts / 1000.0) * (total_seconds / 3600.0) * usd_per_kwh


def cost_per_hour(watts: float, usd_per_kwh: float = DEFAULT_USD_PER_KWH) -> float:
    return (watts / 1000.0) * usd_per_kwh


def cloud_cost_usd(prompt_tokens: int, completion_tokens: int, ref: dict = None) -> float:
    ref = ref or CLOUD_REFERENCE
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return (prompt_tokens / 1000.0) * ref["in_usd_per_1k"] + (completion_tokens / 1000.0) * ref["out_usd_per_1k"]


def savings_pct(local_usd: float, cloud_usd: float) -> float:
    """Percent saved by running locally; 0..100, negative if local is more expensive."""
    if cloud_usd <= 0:
        return 0.0
    return (1.0 - local_usd / cloud_usd) * 100.0


def fmt_usd(x: float) -> str:
    if x >= 1.0:
        return "$%.2f" % x
    if x >= 0.01:
        return "$%.4f" % x
    if x == 0:
        return "$0.00"
    return "$<0.0001"
