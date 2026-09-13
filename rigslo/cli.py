"""Command-line interface.

    rigslo list-models | list-rigs
    rigslo plan --rig rtx-4090 --model llama3-8b --quant int4 --seq 8192
    rigslo matrix --rig rtx-4090 --models llama3-8b,qwen2.5-32b
    rigslo probe --url http://127.0.0.1:8999 --model mock-32b
    rigslo report --rig rtx-4090 --models llama3-8b,qwen2.5-7b --out report.html
    rigslo export --rig rtx-4090 --models llama3-8b --out plans.json
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .models import get, all_models
from .rigs import get_rig, all_rigs
from .slo import plan, max_seq_len
from .throughput import decode_toks, prefill_toks
from .cost import cost_per_hour, cost_per_request, cloud_cost_usd, savings_pct, fmt_usd
from .probe import probe
from .report import build_report


def _split(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def _ints(s: str) -> list:
    return [int(x) for x in _split(s)]


def _print_plan(p) -> None:
    fit = "fits" if p.fits else "NO FIT"
    dash = "—"
    print("model        %s (%s)" % (p.model, p.quant))
    print("rig          %s" % p.rig)
    print("context      %d in / %d out, concurrency %d" % (p.seq_len, p.out_tokens, p.concurrency))
    print("vram         %.2f GB  [%s]" % (p.vram_gb, fit))
    if not p.fits:
        print("max seq len  %d (single sequence)" % max_seq_len(get(p.model), get_rig(p.rig)))
        return
    print("decode       %.1f tok/s" % p.decode_toks)
    print("prefill      %.1f tok/s" % p.prefill_toks)
    print("ttft         %s" % (("%.0f ms" % p.ttft_ms) if p.ttft_ms < 1000 else ("%.2f s" % (p.ttft_ms / 1000))))
    print("p50 / p90    %s / %s" % (_ms(p.p50_ms), _ms(p.p90_ms)))
    print("p90 / p99    %s / %s" % (_ms(p.p90_ms), _ms(p.p99_ms)))
    print("max sessions %d   queue factor %.2f" % (p.max_sessions, p.queue_factor))
    print("throughput   %.1f requests/h" % p.requests_per_hour)


def _ms(ms) -> str:
    if ms == float("inf"):
        return "∞"
    if ms < 1000:
        return "%.0f ms" % ms
    return "%.2f s" % (ms / 1000.0)


def _reconfigure_utf8() -> None:
    """Windows consoles default to cp1252; force UTF-8 for ✔/✘/∞ markers."""
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (stream.encoding or "").lower()
            if enc not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - non-reconfigurable stream
            pass


def main(argv=None) -> int:
    _reconfigure_utf8()
    ap = argparse.ArgumentParser(prog="rigslo", description="Inference capacity planner for local LLM rigs.")
    ap.add_argument("--version", action="version", version="rigslo %s" % __version__)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list-models", help="list the built-in model catalog")
    sub.add_parser("list-rigs", help="list the built-in rig catalog")

    p = sub.add_parser("plan", help="plan one model on one rig")
    p.add_argument("--rig", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--quant", default=None, help="fp16|bf16|int8|int4 (default: model default)")
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--out-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--kv-quant", default="fp16")
    p.add_argument("--calibration", type=float, default=1.0)
    p.add_argument("--json", action="store_true", help="emit JSON")

    m = sub.add_parser("matrix", help="fit grid: models x contexts")
    m.add_argument("--rig", required=True)
    m.add_argument("--models", required=True, help="comma-separated catalog names")
    m.add_argument("--seqs", default="2048,8192,32768")

    pr = sub.add_parser("probe", help="measure a live OpenAI-compatible endpoint")
    pr.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8999/v1 or http://127.0.0.1:11434")
    pr.add_argument("--model", required=True)
    pr.add_argument("--api-key", default="sk-local")
    pr.add_argument("--out-tokens", type=int, default=64)
    pr.add_argument("--timeout", type=float, default=120.0)

    r = sub.add_parser("report", help="multi-model HTML capacity report")
    r.add_argument("--rig", required=True)
    r.add_argument("--models", required=True, help="comma-separated catalog names")
    r.add_argument("--seqs", default="8192")
    r.add_argument("--out-tokens", type=int, default=256)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--usd-per-kwh", type=float, default=0.15)
    r.add_argument("--calibration", type=float, default=1.0)
    r.add_argument("--out", default="rigslo_report.html")

    e = sub.add_parser("export", help="export plans as JSON")
    e.add_argument("--rig", required=True)
    e.add_argument("--models", required=True)
    e.add_argument("--seqs", default="2048,8192,32768")
    e.add_argument("--out", default="rigslo_plans.json")

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1

    if args.cmd == "list-models":
        for mm in all_models():
            print("%-18s %5.1fB act %5.1fB  L%02d kv%02d hd%3d ctx%7d  default %s"
                      % (mm.name, mm.params_b, mm.active_params_b, mm.n_layers, mm.n_kv_heads,
                         mm.head_dim, mm.context, mm.default_quant))
        return 0

    if args.cmd == "list-rigs":
        for rr in all_rigs():
            print("%-10s %-22s %4.0f GB  %6.0f GB/s  %5.0f TFLOPS  %4.0f W"
                % (rr.name, rr.label, rr.vram_gb, rr.mem_bw_gbps, rr.tflops_fp16, rr.base_watts))
        return 0

    if args.cmd == "plan":
        rig = get_rig(args.rig)
        mm = get(args.model)
        p = plan(mm, rig, quant=args.quant, seq_len=args.seq, out_tokens=args.out_tokens,
                 concurrency=args.concurrency, kv_quant=args.kv_quant, calibration=args.calibration)
        if args.json:
            print(json.dumps(p.to_dict(), indent=2))
        else:
            _print_plan(p)
        return 0

    if args.cmd == "matrix":
        from .models import vram_bytes
        rig = get_rig(args.rig)
        seqs = _ints(args.seqs)
        models = _split(args.models)
        header = "%-18s %10s" % ("model", "max seq") + "".join("%10d" % s for s in seqs)
        print(header)
        for name in models:
            mm = get(name)
            line = "%-18s %10d" % (name, max_seq_len(mm, rig))
            for s in seqs:
                f = vram_bytes(mm, s, mm.default_quant) + int(0.5e9) <= int(rig.vram_gb * 1e9)
                line += "%10s" % ("✔" if f else "✘")
            print(line)
        return 0

    if args.cmd == "probe":
        base = args.url
        if base.endswith("/v1"):
            base = base[:-3]
        res = probe(base, args.model, api_key=args.api_key,
                    completion_tokens=args.out_tokens, timeout=args.timeout)
        if not res.ok:
            print("probe failed: %s" % res.error)
            return 1
        print("model            %s" % res.model)
        print("ttft             %.0f ms" % res.ttft_ms)
        print("decode           %.1f tok/s" % res.decode_toks)
        print("tokens           %d in / %d out" % (res.prompt_tokens, res.completion_tokens))
        print("wall             %.0f ms" % res.wall_ms)
        return 0

    if args.cmd == "report":
        build_report(args.rig, _split(args.models), seqs=_ints(args.seqs),
                     out_tokens=args.out_tokens, concurrency=args.concurrency,
                     usd_per_kwh=args.usd_per_kwh, calibration=args.calibration, out_path=args.out)
        print("wrote %s" % args.out)
        return 0

    if args.cmd == "export":
        rig = get_rig(args.rig)
        seqs = _ints(args.seqs)
        out = []
        for s in seqs:
            for name in _split(args.models):
                p = plan(get(name), rig, seq_len=s)
                out.append(p.to_dict())
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rig": rig.name, "plans": out}, f, indent=2)
        print("wrote %s (%d plans)" % (args.out, len(out)))
        return 0

    ap.print_help()
    return 1
