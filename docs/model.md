# RigSLO model assumptions

RigSLO is an *analytic* planner, not a simulator. Every number it prints
traces to a named constant in `rigslo/*.py`, and every assumption is
documented here so you can challenge one number at a time.

## 1. VRAM arithmetic

```
weights_bytes = active_params × bytes_per_quant + 128 MiB
kvcache_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × kv_bytes
vram_bytes    = weights_bytes + kvcache_bytes
fits          ⇔ vram_bytes + 0.5 GB headroom ≤ rig VRAM
```

* `bytes_per_quant`: fp16/bf16 = 2.0, int8 = 1.0, int4 = 0.5.
  (4-bit kernels often store scales in higher precision; 0.5 B/weight is the
  standard working approximation.)
* 128 MiB fixed overhead covers CUDA context, cuBLAS/cuDNN workspaces, and
  activation buffers. Small models are slightly pessimistic, large models
  nearly exact.
* KV cache uses `n_kv_heads` (MQA=1, GQA=8, etc.) — the actual number of
  stored KV vectors, not query heads.
* MoE models price the **active** parameter count per token (the weight that
  is *read* each step), while the catalog keeps total params for reference.
  Router/embedding extra memory is folded into the 128 MiB overhead.
* Architectural values (layers, kv heads, head dim) are approximate
  datasheet numbers, centralized in `rigslo/models.py`. Register your own
  exact values with `rigslo.models.register()` if they matter to you.

## 2. Throughput

**Decode (memory-bound).** Each generated token executes one forward pass
that reads every active weight once:

```
tok/s_decode = mem_bw × 0.85 / (weights_bytes + kvcache_bytes(seq))
```

The 0.85 efficiency factor absorbs dequantization, kernel launch overhead,
and Python/engine glue. Real engines land at 0.7–1.0 of the bandwidth roof
for int4 quantizations; the probe mode lets you replace the constant with
your measured ratio.

**Prefill (compute-bound).** A token costs ~2 FLOPs per active parameter:

```
tok/s_prefill = tflops_fp16 × 0.40 / (2 × active_params)
```

0.40 reflects the gap between datasheet tensor-core peak and sustained
effective throughput (precision, occupancy, batching at seq_len≈prompt).

**Calibration.** `probe` measures a real endpoint; `calibration =
probe_toks / analytic_toks` multiplies both decode and prefill. One probe
anchors the whole model.

## 3. SLOs

```
ttft = seq_in / tok/s_prefill
t    = ttft + seq_out / tok/s_decode        (= p50)
p90  = p50 × 1.15
p99  = p50 × 1.35 × queue_factor
queue_factor = 1 + max(0, concurrency − max_sessions) / max_sessions
```

* No sampling: percentiles are analytic, so runs are byte-reproducible.
* `max_sessions` = KV budget (VRAM − weights − headroom) ÷ per-sequence KV
  size, floored. Above that, requests queue and the linear penalty applies.
  (An M/M/1 model would be more precise; the linear form is more legible
  and conservative at the low concurrency ranges local rigs actually see.)
* p50 throughput in requests/hour uses the unqueued p50; queue factor
  already degrades p90/p99.

## 4. Power & cost

* `base_watts` is **whole-system** sustained draw under load (GPU + CPU +
  memory + fans), approximate from vendor/teardown data, not GPU TDP.
* `$ = (W/1000) × (hours) × $/kWh` — no efficiency term; a rig draws what it
  draws.
* The cloud comparator (`CLOUD_REFERENCE`) is a reference *price class*
  (gpt-4o-mini-class list prices), included so local $ can be read against
  a hosted alternative, not as a vendor quote. Update it when prices move.

## 5. What RigSLO deliberately does not model

* Batch > 1 compute efficiency (decode is priced per-request).
* CPU offload (weights partially on RAM) — register a rig with reduced
  effective bandwidth if that is your setup.
* Multi-GPU tensor/pipeline parallelism — register an aggregate rig.
* Thermal throttling over long sessions.

None of these are unknowable; they are scope. The interface (rig catalog +
calibration factor) is where they plug in.
