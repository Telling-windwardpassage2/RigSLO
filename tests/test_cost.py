from __future__ import annotations

import math

import pytest

from rigslo.cost import (cost_per_request, cost_per_hour, cloud_cost_usd,
                         savings_pct, fmt_usd, CLOUD_REFERENCE)


def test_cost_per_request_units():
    # 450 W for 10 s at $0.15/kWh = 450/1000 * (10/3600) * 0.15
    expect = (450 / 1000.0) * (10 / 3600.0) * 0.15
    assert abs(cost_per_request(450, 10, 0.15) - expect) < 1e-15


def test_cost_per_hour():
    assert abs(cost_per_hour(450, 0.15) - 0.0675) < 1e-12


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        cost_per_request(-1, 10)
    with pytest.raises(ValueError):
        cost_per_request(100, -5)
    with pytest.raises(ValueError):
        cloud_cost_usd(-1, 10)


def test_cloud_cost_reference():
    ref = CLOUD_REFERENCE
    expect = (8000 / 1000.0) * ref["in_usd_per_1k"] + (512 / 1000.0) * ref["out_usd_per_1k"]
    assert abs(cloud_cost_usd(8000, 512) - expect) < 1e-12


def test_cloud_cost_custom_reference():
    ref = {"name": "x", "in_usd_per_1k": 1.0, "out_usd_per_1k": 2.0}
    assert abs(cloud_cost_usd(1000, 1000, ref) - 3.0) < 1e-12


def test_savings_pct():
    assert savings_pct(0.01, 0.10) == 90.0
    assert savings_pct(0.10, 0.05) == -100.0
    assert savings_pct(1.0, 0.0) == 0.0  # degenerate denominator


def test_fmt_usd_tiers():
    assert fmt_usd(5.0) == "$5.00"
    assert fmt_usd(0.05) == "$0.0500"
    assert fmt_usd(0.0) == "$0.00"
    assert fmt_usd(1e-9) == "$<0.0001"


def test_local_cheaper_than_cloud_for_small_model():
    # 450 W rig, ~3 s request, $0.15/kWh vs cloud for 8k in / 256 out
    local = cost_per_request(450, 3.0, 0.15)
    cloud = cloud_cost_usd(8192, 256)
    assert local < cloud
    assert savings_pct(local, cloud) > 50.0
