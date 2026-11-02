# BELLS vs Strata/Basalt review

Source reports:

- `C:\Users\Daniel\Downloads\analize\optimizing-llamacpp-bells-performance`
- `C:\Users\Daniel\Downloads\analize\analyze-llm-offloading-methods`

The reports compare BELLS with Strata and Basalt, mostly from source inspection.
Numbers from Basalt/NVFP4/sm_120 were not produced on this machine and are not
treated as BELLS benchmarks.

## Hard constraint

BELLS stays universal:

- all MoE architectures, not one model family;
- all GGML quantization types, including future ones;
- no custom weights, learned constants, or predefined slot counts per quant;
- no model-name or quant-name branches on the inference path;
- exact output by default; lossy cache modes remain opt-in.

This rules out the reports' per-model profile seeding, per-quant CPU kernels with
hardcoded tuning, and mixed-precision cache occupancy as default behavior.

## Ideas that fit the constraint

Ordered by expected value and risk:

1. Make a miss non-blocking. Split each ubatch's experts into resident GPU,
   arriving GPU, and host CPU work. The GPU computes resident experts while
   copies land and the CPU computes the remainder. The weighted sum is exact.
   The decision must use runtime residency and measured copy/CPU time, never a
   per-quant constant.
2. Overlap copies with compute. Issue copies on the existing copy stream and
   gate only the consumers with events. This is a scheduling change, not a
   tensor-type change.
3. Predict the next layer from the current layer's router/activations rather
   than per-model co-occurrence tables. The predictor only changes transfer
   timing; the real router remains authoritative.
4. Global hot-set ranking with decay. Today each layer evicts independently.
   A single ranked hot set with a short adaptation cadence is quant-agnostic.
   Avoid long cadences: the report measured that slow adaptation hurts more
   than the swap volume helps.
5. Elastic VRAM between KV, expert cache, and optional MTP head. Use the
   existing VMM pool, grow/shrink on demand, and keep a bounded headroom.
6. Host tiers that are data-driven: mmap page cache, optional pinned RAM
   complement, and explicit readahead. No per-model or per-quant weights.
7. Prompt path: per-expert staging ring with a threaded issuer and
   router-driven copy-ahead. Keep the existing prompt-staging flags as fallback.
8. CPU side: one pinned worker per physical core, generic ggml quantized
   kernels, and runtime-measured GPU/CPU split. Do not add a separate kernel per
   quantization type.
9. MTP-aware transfer scheduling: protect draft-window slots and amortize
   transfers across the verify window.
10. Batched DMA only when available and only after a runtime capability probe;
    always keep a per-tensor fallback.

## Ideas rejected for BELLS

- Per-model expert profiles shipped in the repo.
- Per-quant slot counts, thresholds, or learned weights.
- A separate hand-written CPU kernel table for selected quants.
- Mixed-precision cache occupancy as a default.
- Any lossy routing or expert dropping.
- Copying Basalt kernel work that requires sm_120/NVFP4.

## Upstream merge note

The fork was 662 commits behind `ggml/master` and introduced `llama_moe_cache`,
`hdmd`, and changed graph/context signatures. The merge was resolved on branch
`upstream-latest`; the CUDA build targets `75;86`, with `exl3.cu` restricted to
sm_86. A Q4 BELLS smoke test measured 26.95 tok/s on 32 slots, within the prior
26.9-30.4 range. No per-quant BELLS policy was added.
