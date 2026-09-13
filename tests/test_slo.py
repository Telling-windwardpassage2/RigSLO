from __future__ import annotations

import math

import pytest

from rigslo.models import get
from rigslo.rigs import get_rig
from rigslo.slo import plan, max_seq_len, P90_MULT, P99_MULT


def test_plan_shape_and_determinism():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    a = plan(m, rig, seq_len=8192, out_tokens=256)
    b = plan(m, rig, seq_len=8192, out_tokens=256)
    assert a == b
    assert a.model == "llama3-8b"
    assert a.rig == "rtx-4090"


def test_percentile_ordering():
    p = plan(get("llama3-8b"), get_rig("rtx-4090"), seq_len=4096, out_tokens=128)
    assert p.p50_ms < p.p90_ms < p.p99_ms
    assert abs(p.p90_ms - p.p50_ms * P90_MULT) < 1e-9


def test_ttft_grows_with_prompt():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    short = plan(m, rig, seq_len=512, out_tokens=128)
    long = plan(m, rig, seq_len=16384, out_tokens=128)
    assert long.ttft_ms > short.ttft_ms
    assert abs(long.ttft_ms / short.ttft_ms - 32.0) < 1e-6


def test_generation_time_proportional_to_output():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    a = plan(m, rig, seq_len=1024, out_tokens=64)
    b = plan(m, rig, seq_len=1024, out_tokens=512)
    assert abs((b.p50_ms - b.ttft_ms) / (a.p50_ms - a.ttft_ms) - 8.0) < 1e-6


def test_max_sessions_shrink_with_seq_len():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    small = plan(m, rig, seq_len=2048)
    big = plan(m, rig, seq_len=32768)
    assert small.max_sessions > big.max_sessions > 0


def test_queue_factor_idle_and_loaded():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    idle = plan(m, rig, seq_len=2048, concurrency=1)
    assert idle.queue_factor == 1.0
    loaded = plan(m, rig, seq_len=2048, concurrency=1000)
    assert loaded.queue_factor > 1.0
    assert loaded.p99_ms > idle.p99_ms


def test_no_fit_gives_inf_when_concurrent():
    rig = get_rig("rtx-3060")
    m = get("llama3-70b")
    p = plan(m, rig, seq_len=2048, concurrency=2)
    assert not p.fits
    # weights alone don't fit -> 0 sessions -> infinite queue at c>1
    assert p.max_sessions == 0
    assert p.p99_ms == float("inf")
    assert p.requests_per_hour == 0.0


def test_single_request_never_queues():
    p = plan(get("llama3-70b"), get_rig("rtx-3060"), seq_len=2048, concurrency=1)
    assert p.queue_factor == 1.0


def test_max_seq_len_bounds():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    mx = max_seq_len(m, rig)
    assert mx > 0
    # by construction: mx fits, mx+1 does not
    from rigslo.models import vram_bytes
    assert vram_bytes(m, mx, m.default_quant) + int(0.5e9) <= int(rig.vram_gb * 1e9)
    assert vram_bytes(m, mx + 1, m.default_quant) + int(0.5e9) > int(rig.vram_gb * 1e9)


def test_max_seq_len_zero_when_weights_dont_fit():
    assert max_seq_len(get("llama3-70b"), get_rig("rtx-3060")) == 0


def test_invalid_arguments():
    rig = get_rig("rtx-4090")
    m = get("llama3-8b")
    with pytest.raises(ValueError):
        plan(m, rig, seq_len=0)
    with pytest.raises(ValueError):
        plan(m, rig, out_tokens=0)
    with pytest.raises(ValueError):
        plan(m, rig, concurrency=0)


def test_plan_to_dict_roundtrip():
    p = plan(get("llama3-8b"), get_rig("rtx-4090"))
    d = p.to_dict()
    assert isinstance(d, dict)
    assert d["model"] == "llama3-8b"
    assert all(isinstance(v, (int, float, str, bool)) for v in d.values())
