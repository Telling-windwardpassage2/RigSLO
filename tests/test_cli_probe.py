from __future__ import annotations

import subprocess
import sys

import pytest

from mock_server import start, stop
from rigslo.probe import probe


@pytest.fixture
def server():
    srv, url = start()
    yield url
    stop(srv)


def test_probe_cli_success(server, capsys):
    # run_cli in-process so we can share the fixture
    from rigslo.cli import main
    rc = main(["probe", "--url", server, "--model", "mock-7b", "--out-tokens", "32"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mock-7b" in out
    assert "ttft" in out
    assert "decode" in out


def test_probe_cli_v1_suffix(server, capsys):
    from rigslo.cli import main
    rc = main(["probe", "--url", server + "/v1", "--model", "mock-7b"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ttft" in out


def test_probe_cli_failure(capsys):
    from rigslo.cli import main
    rc = main(["probe", "--url", "http://127.0.0.1:1", "--model", "x", "--timeout", "1.0"])
    err = capsys.readouterr().out
    assert rc == 1
    assert "probe failed" in err
