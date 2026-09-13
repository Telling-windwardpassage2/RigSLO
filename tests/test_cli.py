from __future__ import annotations

import json
import subprocess
import sys

import pytest


def run_cli(*args):
    # The CLI emits UTF-8 (✔/✘ markers); decode as such regardless of locale.
    return subprocess.run([sys.executable, "-m", "rigslo", *args],
                          capture_output=True, text=True, encoding="utf-8", cwd=str(
                              __import__("pathlib").Path(__file__).parent.parent))


def test_list_models():
    r = run_cli("list-models")
    assert r.returncode == 0
    assert "llama3-8b" in r.stdout
    assert "qwen2.5-32b" in r.stdout


def test_list_rigs():
    r = run_cli("list-rigs")
    assert r.returncode == 0
    assert "rtx-4090" in r.stdout
    assert "m3-max" in r.stdout


def test_plan_text_output():
    r = run_cli("plan", "--rig", "rtx-4090", "--model", "llama3-8b", "--quant", "int4",
              "--seq", "8192")
    assert r.returncode == 0
    assert "fits" in r.stdout
    assert "decode" in r.stdout
    assert "p50 / p90" in r.stdout


def test_plan_json_output():
    r = run_cli("plan", "--rig", "rtx-4090", "--model", "llama3-8b", "--json")
    assert r.returncode == 0
    d = json.loads(r.stdout)
    assert d["model"] == "llama3-8b"
    assert d["fits"] is True
    assert d["p99_ms"] > d["p50_ms"]


def test_plan_no_fit_still_exits_zero():
    r = run_cli("plan", "--rig", "rtx-3060", "--model", "llama3-70b", "--json")
    assert r.returncode == 0
    d = json.loads(r.stdout)
    assert d["fits"] is False


def test_plan_unknown_model_fails():
    r = run_cli("plan", "--rig", "rtx-4090", "--model", "nope-1b")
    assert r.returncode != 0


def test_matrix():
    r = run_cli("matrix", "--rig", "rtx-4090", "--models", "llama3-8b,qwen2.5-32b",
                 "--seqs", "2048,32768")
    assert r.returncode == 0
    assert "llama3-8b" in r.stdout
    assert "✔" in r.stdout


def test_export_json(tmp_path):
    out = str(tmp_path / "plans.json")
    r = run_cli("export", "--rig", "rtx-4090", "--models", "llama3-8b,mistral-7b",
              "--seqs", "2048,8192", "--out", out)
    assert r.returncode == 0
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["rig"] == "rtx-4090"
    assert len(d["plans"]) == 4


def test_report_cli(tmp_path):
    out = str(tmp_path / "r.html")
    r = run_cli("report", "--rig", "rtx-4090", "--models", "llama3-8b,qwen2.5-7b",
                      "--out", out)
    assert r.returncode == 0
    content = open(out, encoding="utf-8").read()
    assert "Inference Capacity Report" in content


def test_no_args_shows_help():
    r = run_cli()
    assert r.returncode == 1
    assert "usage" in (r.stdout + r.stderr).lower()


def test_version():
    r = run_cli("--version")
    assert r.returncode == 0
    assert "rigslo" in r.stdout
