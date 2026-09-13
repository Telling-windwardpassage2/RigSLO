from __future__ import annotations

import pytest

from rigslo.models import (Model, CATALOG, get, register, all_models, weights_bytes,
                           kvcache_bytes, vram_bytes, fits, BYTES_PER_QUANT)


def test_quant_bytes_table():
    assert BYTES_PER_QUANT["fp16"] == 2.0
    assert BYTES_PER_QUANT["bf16"] == 2.0
    assert BYTES_PER_QUANT["int8"] == 1.0
    assert BYTES_PER_QUANT["int4"] == 0.5


def test_weights_bytes_scale_with_quant():
    m = get("llama3-8b")
    fp16 = weights_bytes(m, "fp16")
    int4 = weights_bytes(m, "int4")
    assert int4 < fp16
    # int4 is roughly half of fp16 (overhead makes it slightly more than 2x cheaper)
    assert fp16 / int4 > 1.8


def test_weights_bytes_unknown_quant():
    with pytest.raises(ValueError):
        weights_bytes(get("llama3-8b"), "int2")


def test_kvcache_proportional_to_seq():
    m = get("llama3-8b")
    a = kvcache_bytes(m, 1024)
    b = kvcache_bytes(m, 2048)
    assert b == 2 * a
    assert kvcache_bytes(m, 1024, "int8") == kvcache_bytes(m, 1024, "fp16") // 2


def test_kvcache_requires_positive_seq():
    with pytest.raises(ValueError):
        kvcache_bytes(get("llama3-8b"), 0)


def test_vram_is_weights_plus_kv():
    m = get("llama3-8b")
    assert vram_bytes(m, 4096, "int4", "fp16") == weights_bytes(m, "int4") + kvcache_bytes(m, 4096, "fp16")


def test_gqa_models_have_smaller_kv_than_mha():
    # mistral-7b is MQA (1 kv head) vs codellama-7b (32 kv heads), same depth class
    mq = kvcache_bytes(get("mistral-7b"), 8192)
    mha = kvcache_bytes(get("codellama-7b"), 8192)
    assert mq < mha


def test_moe_uses_active_params_for_weights():
    mx = get("mixtral-8x7b")
    assert mx.active_params_b < mx.params_b
    assert weights_bytes(mx, "int4") < weights_bytes(get("qwen2.5-32b"), "int4")


def test_get_unknown_model():
    with pytest.raises(KeyError):
        get("nope-1b")


def test_register_extends_catalog():
    n = len(CATALOG)
    register(Model("test-1b", "test", 1.0, 1.0, 4, 1, 64, 4096))
    assert len(CATALOG) == n + 1
    assert get("test-1b").params_b == 1.0
    del CATALOG["test-1b"]


def test_all_models_sorted_by_params():
    ms = all_models()
    sizes = [m.params_b for m in ms]
    assert sizes == sorted(sizes)


def test_fits_boundaries():
    m = get("llama3-8b")
    # 6 GB card: 8B int4 (~4.7 GB + headroom) fits at 8k but NOT at 32k
    assert fits(m, 6 * 1e9, 8192, "int4")
    assert not fits(m, 6 * 1e9, 32768, "int4")
    # 12 GB card: fits comfortably
    assert fits(m, 12 * 1e9, 8192, "int4")
    # fp16 at 32k crosses the 12 GB boundary
    assert not fits(m, 12 * 1e9, 32768, "fp16")


def test_70b_needs_big_card():
    m = get("llama3-70b")
    assert not fits(m, 24 * 1e9, 4096, "int8")
    assert fits(m, 48 * 1e9, 4096, "int4")
