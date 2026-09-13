from __future__ import annotations

import pytest

from rigslo.models import get
from rigslo.rigs import get_rig, Rig
from rigslo.throughput import decode_toks, prefill_toks, DECODE_EFF, PREFILL_EFF


def test_decode_monotonic_in_bandwidth():
    m = get("llama3-8b")
    fast = get_rig("rtx-4090")
    slow = get_rig("rtx-3060")
    assert decode_toks(m, fast) > decode_toks(m, slow)


def test_decode_monotonic_in_quantization():
    m = get("llama3-8b")
    rig = get_rig("rtx-4090")
    assert decode_toks(m, rig, quant="int4") > decode_toks(m, rig, quant="int8") > decode_toks(m, rig, quant="fp16")


def test_decode_decreases_with_seq_len():
    m = get("llama3-8b")
    rig = get_rig("rtx-4090")
    assert decode_toks(m, rig, seq_len=1024) > decode_toks(m, rig, seq_len=8192)


def test_prefill_monotonic_in_compute():
    m = get("llama3-8b")
    assert prefill_toks(m, get_rig("rtx-5090")) > prefill_toks(m, get_rig("rtx-4090"))
    assert prefill_toks(m, get_rig("rtx-4090")) > prefill_toks(m, get_rig("rtx-3060"))
    assert prefill_toks(m, get_rig("m3-max")) < prefill_toks(m, get_rig("rtx-4090"))


def test_prefill_scales_inverse_to_params():
    rig = get_rig("rtx-4090")
    small = prefill_toks(get("llama3-8b"), rig)
    big = prefill_toks(get("llama3-70b"), rig)
    ratio = small / big
    assert 6.0 < ratio < 10.0  # ~70/8, minus efficiency noise


def test_calibration_scales_linearly():
    m = get("llama3-8b")
    rig = get_rig("rtx-4090")
    base = decode_toks(m, rig, calibration=1.0)
    assert abs(decode_toks(m, rig, calibration=0.5) - base / 2) < 1e-9


def test_bad_calibration_rejected():
    with pytest.raises(ValueError):
        decode_toks(get("llama3-8b"), get_rig("rtx-4090"), calibration=0.0)


def test_custom_rig():
    rig = Rig("test", "Test Rig", 8, 500.0, 50.0, 200)
    t = decode_toks(get("llama3-8b"), rig)
    assert t > 0
    assert t < decode_toks(get("llama3-8b"), get_rig("rtx-4090"))


def test_efficiency_constants_documented():
    assert 0.0 < DECODE_EFF <= 1.0
    assert 0.0 < PREFILL_EFF <= 1.0
