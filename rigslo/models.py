"""Built-in model catalog and VRAM arithmetic.

Every architectural value here is an *approximate* datasheet number, kept in
one place so the arithmetic is auditable. See ``docs/model.md`` for the full
assumption list. The catalog is plain data on purpose: extend it at runtime
with :func:`register`.

Quantization bytes-per-weight:
    fp16/bf16 = 2.0, int8 = 1.0, int4 = 0.5
plus a fixed 128 MiB overhead for runtime buffers (CUDA context, workspace).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

BYTES_PER_QUANT: Dict[str, float] = {"fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}
FIXED_OVERHEAD_BYTES = int(128 * 1024 * 1024)  # 128 MiB


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    params_b: float        # total parameters, billions
    active_params_b: float # parameters active per token (MoE: less than total)
    n_layers: int
    n_kv_heads: int        # 1 = MQA, < n_heads = GQA
    head_dim: int
    context: int           # trained context length
    default_quant: str = "int4"

    @property
    def weights_base_bytes(self) -> float:
        return self.active_params_b * 1e9


def weights_bytes(m: Model, quant: str = "fp16") -> int:
    """Weight footprint for one quantization, including runtime overhead."""
    if quant not in BYTES_PER_QUANT:
        raise ValueError("unknown quant %r; choose from %s" % (quant, sorted(BYTES_PER_QUANT)))
    return int(m.weights_base_bytes * BYTES_PER_QUANT[quant]) + FIXED_OVERHEAD_BYTES


def kvcache_bytes(m: Model, seq_len: int, quant: str = "fp16") -> int:
    """K+V cache for a single sequence of ``seq_len`` tokens.

    2 (K and V) x layers x kv_heads x head_dim x seq_len x bytes-per-weight.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    q = BYTES_PER_QUANT.get(quant, 2.0)
    return int(2 * m.n_layers * m.n_kv_heads * m.head_dim * seq_len * q)


def vram_bytes(m: Model, seq_len: int, quant: str = "fp16", kv_quant: str = "fp16") -> int:
    return weights_bytes(m, quant) + kvcache_bytes(m, seq_len, kv_quant)


def fits(m: Model, rig_vram_bytes: int, seq_len: int, quant: str = "fp16",
         kv_quant: str = "fp16", overhead_gb: float = 0.5) -> bool:
    """True when weights + one sequence's KV cache fit with headroom."""
    return vram_bytes(m, seq_len, quant, kv_quant) + int(overhead_gb * 1e9) <= rig_vram_bytes


CATALOG: Dict[str, Model] = {
    "llama3-8b":        Model("llama3-8b",        "llama3",   8.0,  8.0,  32,  8, 128, 128000),
    "llama3-70b":       Model("llama3-70b",       "llama3",   70.0, 70.0, 80,  8, 128, 128000),
    "qwen2.5-7b":       Model("qwen2.5-7b",       "qwen2.5",  7.6,  7.6,  28,  4, 128, 128000),
    "qwen2.5-14b":      Model("qwen2.5-14b",      "qwen2.5",  14.8, 14.8, 48,  4, 128, 128000),
    "qwen2.5-32b":      Model("qwen2.5-32b",      "qwen2.5",  32.8, 32.8, 64,  8, 128, 128000),
    "mistral-7b":       Model("mistral-7b",       "mistral",  7.2,  7.2,  32,  1, 128, 32000),
    "mixtral-8x7b":     Model("mixtral-8x7b",     "mixtral",  46.7, 12.9, 32,  1, 128, 32000),
    "gemma2-9b":        Model("gemma2-9b",        "gemma2",   9.2,  9.2,  42,  8, 256, 8000),
    "gemma2-27b":       Model("gemma2-27b",       "gemma2",   27.0, 27.0, 62, 16, 256, 8000),
    "phi4":             Model("phi4",             "phi",      14.0, 14.0, 40, 10, 128, 16000),
    "deepseek-r1-7b":   Model("deepseek-r1-7b",   "deepseek-r1-distill", 7.6,  7.6,  28,  4, 128, 128000),
    "deepseek-r1-14b":  Model("deepseek-r1-14b",  "deepseek-r1-distill", 14.8, 14.8, 48,  4, 128, 128000),
    "deepseek-r1-32b":  Model("deepseek-r1-32b",  "deepseek-r1-distill", 32.8, 32.8, 64,  8, 128, 128000),
    "command-r":        Model("command-r",        "command-r", 35.0, 35.0, 40,  4, 128, 128000),
    "codellama-7b":     Model("codellama-7b",     "codellama", 7.0,  7.0,  32, 32, 128, 128000),
    "codellama-34b":    Model("codellama-34b",    "codellama", 34.0, 34.0, 48, 40, 128, 128000),
}


def get(name: str) -> Model:
    if name not in CATALOG:
        raise KeyError("unknown model %r; known: %s" % (name, sorted(CATALOG)))
    return CATALOG[name]


def register(m: Model) -> None:
    """Add/override a catalog entry (e.g. for an uncataloged local model)."""
    CATALOG[m.name] = m


def all_models() -> List[Model]:
    return sorted(CATALOG.values(), key=lambda m: (m.params_b, m.name))


def by_name(names) -> List[Model]:
    return [get(n) for n in names]
