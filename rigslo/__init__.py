"""RigSLO — inference capacity planning for local LLM rigs.

Zero-dependency (stdlib only). Given a rig (GPU VRAM / bandwidth / compute /
power) and a model (params / architecture / quantization), RigSLO answers:

* does it fit (weights + KV cache + headroom),
* at what speed (analytical decode/prefill tok/s),
* what SLOs to expect (TTFT, total latency p50/p90/p99, max concurrency),
* what it costs (kWh/h, $/h, and cloud-equivalent per request).

A live probe mode (``rigslo probe``) measures a real OpenAI-compatible
endpoint and calibrates the analytical model against it.
"""

__version__ = "0.1.0"

from .models import Model, CATALOG, get, register, all_models, weights_bytes, kvcache_bytes, vram_bytes, fits
from .rigs import Rig, RIGS, get_rig, register_rig, all_rigs
from .throughput import decode_toks, prefill_toks
from .slo import plan, Plan
from .cost import cost_per_request, cost_per_hour, cloud_cost_usd, savings_pct, fmt_usd, CLOUD_REFERENCE

__all__ = [
    "__version__",
    "Model", "CATALOG", "get", "register", "all_models",
    "weights_bytes", "kvcache_bytes", "vram_bytes", "fits",
    "Rig", "RIGS", "get_rig", "register_rig", "all_rigs",
    "decode_toks", "prefill_toks",
    "plan", "Plan",
    "cost_per_request", "cost_per_hour", "cloud_cost_usd", "savings_pct", "fmt_usd", "CLOUD_REFERENCE",
]
