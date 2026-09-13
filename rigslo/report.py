"""Static HTML capacity report — Atelier styling.

One deterministic HTML file: title block, KPI stat strip, decode-throughput
bar chart (inline SVG), setup, main capacity table, model x context fit
matrix, and power/$ table. No external assets, no JS, no timestamps beyond
the caller-supplied (or fixed) generated_ts — byte-identical re-runs.
"""
from __future__ import annotations

import html
import time
from typing import List, Optional, Sequence

from .models import get, vram_bytes
from .rigs import get_rig
from .slo import plan, Plan, max_seq_len
from .cost import cost_per_hour, cost_per_request, cloud_cost_usd, savings_pct, fmt_usd, CLOUD_REFERENCE

_OVERHEAD_GB = 0.5
_FIT_SEQS = [2048, 4096, 8192, 16384, 32768, 65536]

_CSS = (
    "body{background:#DFDFDF;margin:0;padding:32px 20px;font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;"
    "color:#252524;line-height:1.5;font-size:14px}"
    ".page{max-width:1060px;margin:0 auto}"
    "h1{font-family:Georgia,'Times New Roman',serif;font-size:28px;font-weight:700;margin:0 0 4px;color:#252524}"
    ".sub{color:#676662;font-size:13px;margin-bottom:22px}"
    "h2{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;font-size:11px;font-weight:600;"
    "letter-spacing:0.10em;text-transform:uppercase;color:#94938C;margin:0;padding:16px 20px 8px}"
    ".card{background:#DFDFDB;border:1px solid #DCDAD1;border-radius:8px;margin-bottom:20px;overflow:hidden}"
    ".card-body{padding:0 20px 16px}"
    ".stat-strip{display:flex;background:#DFDFDB;border:1px solid #DCDAD1;border-radius:8px;margin-bottom:20px}"
    ".stat{flex:1;padding:14px 18px}"
    ".stat+.stat{border-left:1px solid #DCDAD1}"
    ".caps{font-size:10px;font-weight:600;letter-spacing:0.10em;text-transform:uppercase;color:#94938C;display:block;margin-bottom:5px}"
    ".numeral{font-family:Georgia,'Times New Roman',serif;font-size:24px;font-weight:700;color:#252524;"
    "font-variant-numeric:tabular-nums;line-height:1.15}"
    ".delta{font-size:11px;color:#676662;margin-top:3px}"
    "table{border-collapse:collapse;width:100%;font-size:13px}"
    "th{font-size:10px;font-weight:600;letter-spacing:0.10em;text-transform:uppercase;color:#94938C;"
    "background:#DFDFDF;border-bottom:1px solid #D4D2C8;padding:9px 10px;text-align:left;vertical-align:bottom}"
    "td{padding:8px 10px;border-top:1px solid #DCDAD1;vertical-align:top}"
    "tbody tr:first-child td{border-top:0}"
    "td.num{font-family:Georgia,'Times New Roman',serif;font-variant-numeric:tabular-nums;text-align:right;color:#252524}"
    "th.num{text-align:right}"
    ".fit{color:#22AC80;font-weight:700}.nofit{color:#A74221;font-weight:700}"
    ".muted{color:#676662;font-size:12.5px}"
    ".note{padding:4px 20px 16px;font-size:13px;color:#252524}"
    ".note b{font-family:Georgia,'Times New Roman',serif}"
    ".chart{padding:2px 20px 16px}"
    ".chart svg{display:block;width:100%;height:auto}"
    "code{font-family:Consolas,'SF Mono',Menlo,monospace;font-size:12px;background:#DFDFDF;padding:1px 5px;border-radius:3px}"
    "footer{margin-top:4px;padding:12px 4px 0;border-top:1px solid #D4D2C8;color:#94938C;font-size:12px}"
)


def _esc(s) -> str:
    return html.escape(str(s))


def _fmt_ms(ms) -> str:
    if ms == float("inf"):
        return "∞"
    if ms < 1000:
        return "%.0f ms" % ms
    return "%.2f s" % (ms / 1000.0)


def _fmt_toks(t) -> str:
    return "%.1f" % t


def _bar_chart(rows, seq_label: str) -> str:
    """Horizontal bar chart of decode tok/s per model (first context).

    rows: list of (model_name, fits, decode_toks). Deterministic geometry:
    paper gridlines, green bars, brick stubs for no-fit models.
    """
    W, LABEL_W, VALUE_W, ROW_H, TOP = 960, 170, 84, 40, 8
    bar_max = W - LABEL_W - VALUE_W
    vals = [d for (_n, f, d) in rows if f]
    vmax = max(vals) if vals else 1.0
    h = TOP + len(rows) * ROW_H + 18
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" aria-label="decode throughput by model">'
             % (W, h)]
    # gridlines at 0/25/50/75/100% of vmax
    for i in range(5):
        frac = i / 4.0
        x = LABEL_W + bar_max * frac
        tick = int(round(vmax * frac))
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#DCDCDC" stroke-width="1"/>'
                     % (x, TOP - 4, x, TOP + len(rows) * ROW_H))
        parts.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="9" fill="#94938C">%d</text>'
                     % (x, TOP + len(rows) * ROW_H + 12, tick))
    for i, (name, fits, d) in enumerate(rows):
        y = TOP + i * ROW_H
        parts.append('<text x="%d" y="%d" text-anchor="end" font-size="12" fill="#676662">%s</text>'
                     % (LABEL_W - 10, y + 16, _esc(name)))
        if fits:
            bw = max(4.0, d / vmax * bar_max)
            parts.append('<rect x="%d" y="%d" width="%.1f" height="20" rx="2" fill="#22AC80"/>'
                         % (LABEL_W, y, bw))
            parts.append('<text x="%.1f" y="%d" font-family="Georgia,serif" font-size="12" fill="#252524">%s</text>'
                         % (LABEL_W + bw + 8, y + 16, _esc(_fmt_toks(d))))
        else:
            parts.append('<rect x="%d" y="%d" width="26" height="20" rx="2" fill="#D4D2C8"/>'
                         % (LABEL_W, y))
            parts.append('<text x="%d" y="%d" font-size="11" fill="#94938C">no fit</text>'
                         % (LABEL_W + 34, y + 16))
    parts.append("</svg>")
    return "".join(parts)


def build_report(rig_name: str, models: Sequence[str], *, seqs: Sequence[int] = (8192,),
                 out_tokens: int = 256, concurrency: int = 1, usd_per_kwh: float = 0.15,
                 calibration: float = 1.0, out_path: str = None,
                 generated_ts: Optional[int] = None) -> str:
    rig = get_rig(rig_name)
    now = generated_ts if generated_ts is not None else int(time.time())
    ts_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now))

    plans_by_seq = {}
    for s in seqs:
        row = []
        for name in models:
            m = get(name)
            p = plan(m, rig, seq_len=s, out_tokens=out_tokens, concurrency=concurrency,
                     calibration=calibration)
            row.append((m, p))
        plans_by_seq[s] = row

    # KPI strip figures (deterministic: pure functions of the plans)
    all_rows = [(s, m, p) for s in seqs for (m, p) in plans_by_seq[s]]
    fit_rows = [r for r in all_rows if r[2].fits]
    fastest = max(fit_rows, key=lambda r: r[2].decode_toks) if fit_rows else None
    worst = max(fit_rows, key=lambda r: r[2].p99_ms) if fit_rows else None
    usd_h = cost_per_hour(rig.base_watts, usd_per_kwh)

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append("<title>RigSLO — capacity report</title>")
    parts.append("<style>%s</style></head><body><div class=\"page\">" % _CSS)

    parts.append("<h1>RigSLO — Inference Capacity Report</h1>")
    parts.append('<div class="sub">%s · %s · generated %s</div>'
                 % (_esc(rig.label), _esc(", ".join(models)), _esc(ts_str)))

    parts.append('<div class="stat-strip">')
    parts.append('<div class="stat"><span class="caps">Rows fitting</span>'
                 '<div class="numeral">%d / %d</div>'
                 '<div class="delta">across %d context%s</div></div>'
                 % (len(fit_rows), len(all_rows), len(seqs), "" if len(seqs) == 1 else "s"))
    if fastest:
        s_f, m_f, p_f = fastest
        parts.append('<div class="stat"><span class="caps">Fastest decode</span>'
                     '<div class="numeral">%s</div>'
                     '<div class="delta">tok/s · %s @ %d</div></div>'
                     % (_esc(_fmt_toks(p_f.decode_toks)), _esc(m_f.name), s_f))
    else:
        parts.append('<div class="stat"><span class="caps">Fastest decode</span>'
                     '<div class="numeral">—</div><div class="delta">nothing fits</div></div>')
    if worst:
        s_w, m_w, p_w = worst
        parts.append('<div class="stat"><span class="caps">Worst p99</span>'
                     '<div class="numeral">%s</div>'
                     '<div class="delta">%s @ %d</div></div>'
                     % (_esc(_fmt_ms(p_w.p99_ms)), _esc(m_w.name), s_w))
    else:
        parts.append('<div class="stat"><span class="caps">Worst p99</span>'
                     '<div class="numeral">—</div><div class="delta">nothing fits</div></div>')
    parts.append('<div class="stat"><span class="caps">$/h sustained</span>'
                 '<div class="numeral">%s</div>'
                 '<div class="delta">%.0f W whole-system draw</div></div>'
                 % (_esc(fmt_usd(usd_h)), rig.base_watts))
    parts.append("</div>")

    # decode throughput chart (first context = best case per model)
    chart_rows = [(m.name, p.fits, p.decode_toks) for (m, p) in plans_by_seq[seqs[0]]]
    parts.append('<div class="card"><h2>Decode throughput — %d context</h2>' % seqs[0])
    parts.append('<div class="chart">%s</div></div>' % _bar_chart(chart_rows, str(seqs[0])))

    parts.append('<div class="card"><h2>1. Setup</h2>')
    parts.append(
        '<div class="note">Rig: <b>%s</b> — %.0f GB VRAM, %.0f GB/s bandwidth, %.0f TFLOPS FP16, ~%.0f W sustained draw. '
        "Contexts: %s. Output budget: %d tokens, concurrency %d. Calibration factor: %.2f. "
        "Electricity: $%.3f/kWh. Cloud comparator: %s ($%.2f / 1k in, $%.2f / 1k out).</div>"
        % (_esc(rig.label), rig.vram_gb, rig.mem_bw_gbps, rig.tflops_fp16, rig.base_watts,
           _esc(", ".join(str(s) for s in seqs)), out_tokens, concurrency, calibration,
           usd_per_kwh, _esc(CLOUD_REFERENCE["name"]),
           CLOUD_REFERENCE["in_usd_per_1k"], CLOUD_REFERENCE["out_usd_per_1k"]))
    parts.append("</div>")

    parts.append('<div class="card"><h2>2. Capacity table</h2><div class="card-body">')
    parts.append("<table><thead><tr><th>model</th><th>quant</th><th class='num'>seq</th><th class='num'>VRAM</th>"
                 "<th>fit</th><th class='num'>decode tok/s</th><th class='num'>prefill tok/s</th>"
                 "<th class='num'>TTFT</th><th class='num'>p50</th><th class='num'>p90</th><th class='num'>p99</th>"
                 "<th class='num'>max sess</th></tr></thead><tbody>")
    for s in seqs:
        for m, p in plans_by_seq[s]:
            fit = '<span class="fit">✔ fits</span>' if p.fits else '<span class="nofit">✘ no fit</span>'
            dash = "—"
            parts.append(
                "<tr><td>%s</td><td>%s</td><td class='num'>%d</td><td class='num'>%.1f GB</td><td>%s</td>"
                "<td class='num'>%s</td><td class='num'>%s</td><td class='num'>%s</td><td class='num'>%s</td>"
                "<td class='num'>%s</td><td class='num'>%s</td><td class='num'>%d</td></tr>"
                % (_esc(m.name), _esc(p.quant), s, p.vram_gb, fit,
                   _fmt_toks(p.decode_toks) if p.fits else dash,
                   _fmt_toks(p.prefill_toks) if p.fits else dash,
                   _fmt_ms(p.ttft_ms) if p.fits else dash,
                   _fmt_ms(p.p50_ms) if p.fits else dash,
                   _fmt_ms(p.p90_ms) if p.fits else dash,
                   _fmt_ms(p.p99_ms) if p.fits else dash,
                   p.max_sessions))
    parts.append("</tbody></table></div></div>")

    parts.append('<div class="card"><h2>3. Fit matrix (model × context)</h2><div class="card-body">')
    parts.append("<table><thead><tr><th>model</th><th class='num'>max seq</th>" +
                 "".join("<th class='num'>%d</th>" % s for s in _FIT_SEQS) + "</tr></thead><tbody>")
    for name in models:
        m = get(name)
        row = "<tr><td>%s</td><td class='num'>%s</td>" % (_esc(m.name), _esc(str(max_seq_len(m, rig))))
        for s in _FIT_SEQS:
            f = vram_bytes(m, s, m.default_quant) + int(_OVERHEAD_GB * 1e9) <= int(rig.vram_gb * 1e9)
            row += "<td class='num %s'>%s</td>" % ("fit" if f else "nofit", "✔" if f else "✘")
        parts.append(row + "</tr>")
    parts.append("</tbody></table></div></div>")

    parts.append('<div class="card"><h2>4. Power &amp; cost</h2><div class="card-body">')
    parts.append("<table><thead><tr><th>model</th><th class='num'>seq</th><th class='num'>$/h sustained</th>"
                 "<th class='num'>$/request (p50)</th><th class='num'>cloud $/request</th>"
                 "<th class='num'>local savings</th></tr></thead><tbody>")
    for s in seqs:
        for m, p in plans_by_seq[s]:
            if not p.fits:
                continue
            ph = cost_per_hour(rig.base_watts, usd_per_kwh)
            pr = cost_per_request(rig.base_watts, p.p50_ms / 1000.0, usd_per_kwh)
            cl = cloud_cost_usd(s, out_tokens)
            sv = savings_pct(pr, cl)
            parts.append(
                "<tr><td>%s</td><td class='num'>%d</td><td class='num'>%s</td><td class='num'>%s</td>"
                "<td class='num'>%s</td><td class='num'>%.1f%%</td></tr>"
                % (_esc(m.name), s, _esc(fmt_usd(ph)), _esc(fmt_usd(pr)), _esc(fmt_usd(cl)), sv))
    parts.append("</tbody></table></div></div>")

    parts.append('<div class="card"><h2>5. Method</h2>')
    parts.append('<div class="card-body"><p class="muted">Decode: memory-bandwidth-bound, one full weight read plus the active KV cache per '
                 'token (×0.85 efficiency). Prefill: 2 FLOPs/param/token (×0.40 efficiency). Percentiles are analytic '
                 'multipliers on expected time (p90=×1.15, p99=×1.35); queueing adds a linear penalty above KV-cache '
                 'capacity. Power = sustained whole-system draw. All assumptions are listed in <code>docs/model.md</code>.</p></div></div>')
    parts.append("<footer>RigSLO v0.1.0 · deterministic output · © 2026 Adithya N Raj</footer>")
    parts.append("</div></body></html>")

    out = "\n".join(parts)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
    return out
