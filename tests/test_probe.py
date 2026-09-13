from __future__ import annotations

import pytest

from rigslo.probe import probe, calibration_from_probe
from mock_server import start, stop


@pytest.fixture
def server():
    srv, url = start()
    yield url
    stop(srv)


def test_probe_success(server):
    r = probe(server, "mock-7b")
    assert r.ok, r.error
    assert r.ttft_ms > 0
    assert r.decode_toks > 0
    assert r.completion_tokens == 32
    assert r.wall_ms > r.ttft_ms
    assert r.model == "mock-7b"
    # decode rate sanity: 32 tokens over a few ms of tail
    assert r.decode_toks > 1.0


def test_probe_deterministic_tokens(server):
    a = probe(server, "mock-7b")
    b = probe(server, "mock-7b")
    assert a.completion_tokens == b.completion_tokens == 32
    assert a.prompt_tokens == b.prompt_tokens == 64


def test_probe_url_with_v1_suffix(server):
    r = probe(server + "/v1", "mock-7b")
    assert r.ok


def test_probe_server_error():
    srv, url = start()
    srv.fail = True
    try:
        r = probe(url, "mock-7b")
        assert not r.ok
        assert r.error
    finally:
        stop(srv)


def test_probe_unreachable():
    r = probe("http://127.0.0.1:1", "x", timeout=2.0)
    assert not r.ok
    assert r.error


def test_calibration_ratio():
    assert abs(calibration_from_probe(10.0, 20.0) - 0.5) < 1e-12
    with pytest.raises(ValueError):
        calibration_from_probe(10.0, 0.0)


def test_probe_result_dict():
    srv, url = start()
    try:
        r = probe(url, "mock-7b")
        d = r.to_dict()
        assert set(d) >= {"model", "ttft_ms", "decode_toks", "ok"}
    finally:
        stop(srv)
