"""Live calibration probe for OpenAI-compatible endpoints.

Streams one fixed request, measures TTFT (time to first token) and decode
throughput (tokens / second after first token), and returns a
``ProbeResult``. Feed ``calibration`` into :func:`rigslo.slo.plan` to anchor
the analytic model to measured numbers.

Stdlib only; works against Ollama, vLLM, llama.cpp server, LM Studio,
and the LocalBench mock provider.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, asdict

DEFAULT_TIMEOUT = 120.0

_PROBE_PROMPT = ("Answer with exactly the numbers 1 to 40, one per line, nothing else. "
                 "Start now.\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")


@dataclass(frozen=True)
class ProbeResult:
    model: str
    endpoint: str
    ttft_ms: float
    decode_toks: float
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    ok: bool
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _read_sse(body):
    """Yield (is_content_delta, content, done, usage_dict) per SSE chunk."""
    buf = b""
    while True:
        chunk = body.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                yield True
                return
            try:
                obj = json.loads(payload.decode("utf-8", "replace"))
            except Exception:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            usage = obj.get("usage") or {}
            yield (False, content, bool(choice.get("finish_reason")), usage)


def probe(base_url: str, model: str, api_key: str = "sk-local", *,
          completion_tokens: int = 64, timeout: float = DEFAULT_TIMEOUT) -> ProbeResult:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/v1/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": completion_tokens,
        "messages": [{"role": "user", "content": _PROBE_PROMPT}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ttft_ms = None
            completion_tokens_out = 0
            prompt_tokens = 0
            content_chunks = 0
            for ev in _read_sse(resp):
                now_ms = (time.perf_counter() - t0) * 1000.0
                if ev is True:
                    break
                _, content, _done, usage = ev
                if ttft_ms is None and content:
                    ttft_ms = now_ms
                if content:
                    content_chunks += 1
                    completion_tokens_out += 1  # one delta chunk ~= one token in mock/small engines
                if usage.get("completion_tokens"):
                    completion_tokens_out = int(usage["completion_tokens"])
                if usage.get("prompt_tokens"):
                    prompt_tokens = int(usage["prompt_tokens"])
            wall_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # noqa: BLE001 - report any transport error
        return ProbeResult(model, base_url, 0.0, 0.0, 0, 0,
                           (time.perf_counter() - t0) * 1000.0, False, error=repr(e))

    if ttft_ms is None:
        return ProbeResult(model, base_url, 0.0, 0.0, prompt_tokens, completion_tokens_out,
                           wall_ms, False, error="no tokens received")
    tail_s = max((wall_ms - ttft_ms) / 1000.0, 1e-6)
    decode = completion_tokens_out / tail_s
    return ProbeResult(model, base_url, ttft_ms, decode, prompt_tokens,
                       completion_tokens_out, wall_ms, True)


def calibration_from_probe(probe_toks: float, analytic_toks: float) -> float:
    """Ratio to multiply the analytic model by."""
    if analytic_toks <= 0:
        raise ValueError("analytic throughput must be positive")
    return probe_toks / analytic_toks
