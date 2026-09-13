from __future__ import annotations

from conftest import FIXED_TS
from rigslo.models import CATALOG, Model, register
from rigslo.report import build_report


def _html(tmp_report):
    return build_report("rtx-4090", ["llama3-8b", "qwen2.5-32b", "llama3-70b"],
                        seqs=[8192], out_tokens=256, generated_ts=FIXED_TS, out_path=tmp_report)


def test_report_renders_sections(tmp_report):
    html = _html(tmp_report)
    for needle in ["RigSLO — Inference Capacity Report", "1. Setup", "2. Capacity table",
                   "3. Fit matrix", "4. Power &amp; cost", "5. Method"]:
        assert needle in html
    assert html.count("<table") >= 3


def test_report_lists_all_models(tmp_report):
    html = _html(tmp_report)
    for name in ("llama3-8b", "qwen2.5-32b", "llama3-70b"):
        assert name in html


def test_report_fit_and_nofit_markers(tmp_report):
    html = _html(tmp_report)
    assert "✔ fits" in html
    assert "✘ no fit" in html  # 70B on 24 GB


def test_report_max_seq_column(tmp_report):
    html = _html(tmp_report)
    assert "max seq" in html


def test_report_is_deterministic(tmp_report):
    a = build_report("rtx-4090", ["llama3-8b"], generated_ts=FIXED_TS)
    b = build_report("rtx-4090", ["llama3-8b"], generated_ts=FIXED_TS)
    assert a == b


def test_report_written_to_file(tmp_report):
    _html(tmp_report)
    with open(tmp_report, encoding="utf-8") as f:
        content = f.read()
    assert content.startswith("<!DOCTYPE html>")
    assert content.rstrip().endswith("</html>")


def test_report_html_escaping():
    register(Model("we<ird>&", "test", 2.0, 2.0, 8, 1, 64, 4096))
    try:
        html = build_report("rtx-4090", ["we<ird>&"], generated_ts=FIXED_TS)
        assert "we<ird>&" not in html
        assert "we&lt;ird&gt;&amp;" in html
    finally:
        del CATALOG["we<ird>&"]


def test_report_cost_row_present_for_fitting_model(tmp_report):
    html = _html(tmp_report)
    assert "$/h sustained" in html
    assert "local savings" in html
