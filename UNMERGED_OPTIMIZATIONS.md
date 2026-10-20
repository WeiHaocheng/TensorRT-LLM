# Performance work left unmerged

The run that landed this branch also tried the changes below on side
branches and did not merge them. Each branch starts from the code of the
tasks it lists as its base. To take one, merge its branch and run the
accuracy and performance gates again.

## perf-moe-streamed-mlp

- Branch: `exp/nemotron4-mv2-v3-unmerged/perf-moe-streamed-mlp`
- Result: DONE: No gain: nothing lands. The branch is node/perf-moe-fused-mlp's tip, unchanged; the kernels and probes are in
WS/nodes/perf-moe-streamed-mlp/. Accuracy is unaffected: no change landed, and every variant measured gave
latent_moe_experts' output bit for bit.
What was built and measured: perf-moe-fused-mlp's "next design" on top of latent_moe_experts_fused_act.
- Its up+act Triton GEMM (the producer, P SMs) runs concurrently, on a second stream, with a persistent Triton
  down GEMM (the consumer, C SMs). Per-entry counters order them: the producer releases each stored slice; the
  consumer acquires the entry before its TMA reads of h.
- Variant: h in a 48-slot ring with back-pressure, marked L2-persisting through a CUDA access-policy window.
Per call against latent_moe_experts_fused_act, same GPU, CUDA-graph replay, real routing:
- tp1 8192: 1263.7 -> 1285.1 us (+1.7%); 1267.6 (+0.3%) with the persisting ring, whose 48 MB L2 set-aside
  slows fused_act itself by 1.0%.
- tp1 16384: +6.4%.
- dep4, slowest window: +6.3% at 8192/rank, +6.5% at 16384/rank.
- tp4 8192: -1.3%, within noise.
Why: half the SMs makes each GEMM about 1.5x slower, not 2x. So even with h in L2, the pair's floor is about
1000 us against 1160 us run one after the other. That margin is eaten by the cross-CTA release (+8% on the
producer at 76 SMs), start, tail and back-pressure bubbles, and HBM contention. And h leaves L2 before it is
read unless a persisting L2 set-aside is used, which slows every other kernel.
The design that could win is one persistent warp-specialized kernel running both GEMMs from one queue on every
SM: bound about -14% per call at tp1 8K and about 0 at dep4 and 16K. It is a multi-day kernel.
- Base: perf-moe-fused-mlp
- Full report: `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/nemotron4-mv2-v3-ws/task_results/task_result_perf-moe-streamed-mlp.md`

**Goal.** Side task: not merged by default, and its branch is kept for a human. It implements `perf-moe-fused-mlp`'s "next design" for the routed experts at prefill and mixed sizes (`WS/task_results/task_result_perf-moe-fused-mlp.md`, *The full fused MLP*), on top of `latent_moe_experts_fused_act`.
- **What remains.** The intermediate h still goes through HBM, about 180 µs of a 1264 µs call at 8K on tp1. The fully fused MLP cannot fit TMEM, and it cannot win on compute.
- **The design.** Two persistent kernels on two streams in one graph:
  - the up + activation kernel releases a per-entry counter after its slices are stored;
  - the down kernel acquires it before reading that entry's h;
  - with a proxy fence before the TMA reads, grids that fit together, and a ring buffer for h with back-pressure.
  - A better design is welcome.
- **Expected** (stated first, criterion C): -5 to -12% per call. That is about -1.8 to -4.2 ms per 8K prompt step on tp1, +1.5–3.5% tok/s/GPU at the throughput end. The risk is high.
- **Built on `perf-moe-fused-mlp`'s tree.** Keep `latent_moe_experts`' bits if possible; otherwise use bands against its reference.
- **Time box:** about 5 h.

**Definition of done.**
- The entry's tests pass.
- A same-GPU before / after per call against `latent_moe_experts_fused_act` at 8K and 16K (tp1, and a dep4 rank's shapes).
- A `WS/perf_ledger.md` row.
- The result's `summary` gives the change, the speedup and its shapes, and the effect on accuracy, for `UNMERGED_OPTIMIZATIONS.md`.

**Output size (read this).** A tool output over 1 MB ends your session: `report-prep`'s attempt 1 died of `CLIJSONDecodeError('Failed to decode JSON: JSON message exceeded maximum buffer size of 1048576 bytes...')` after doing its work and before writing its result. Never print a whole large file or an unbounded diff (`perf_model.md` is 258 KB; server logs and `sweep.json` dumps run to several MB). Use `head`, `tail`, `grep` and `wc`, and keep every command's output under ~100 KB.

**Waiting (read this).** Your session ends the moment you reply without a tool call. Background commands, Monitors and detached pollers do not resume it; the orchestrator then demands your result, and the task fails unverified. So while any GPU job is queued or running, every reply is a foreground tool call: `timeout 540 bash -c 'until [ -f <sentinel> ]; do sleep 30; done'`, repeated until the sentinel exists, then the next step. The pool is full, and a whole-node job can queue for an hour. `op-cells-16k` and `mtp-identity-dep4` attempt 1 ended this way after writing everything and running nothing.

**GPUs.** One GPU, at `perf` priority.

## side-gdp-replay-tp1

- Branch: `exp/nemotron4-mv2-v3-unmerged/side-gdp-replay-tp1`
- Result: DONE: Side task, not merged: branch node/side-gdp-replay-tp1, d16d337838, on int-tp1's 47fb78639e. At MTP draft length 2,
tp1's GDP verify now runs gdp_replay and the target commits the recurrent state itself, so the engine keeps no
per-position recurrent states.
- Engine change (Python only: +12 lines in _util.py, +38/-3 in mamba_cache_manager.py):
  - A hybrid manager built with model_commits_ssm_states allocates no intermediate_ssm, and its
    update_mamba_states promotes only the conv state.
  - _create_kv_cache_manager sets the flag from the model config's
    extra_attrs["commits_speculative_ssm_states"]. Only tp1 at D = 2 sets that; no other model gets the keyword.
  - The one-model MTP-Eagle worker is unchanged. It already calls update_mamba_states (conv-only for this model
    now) and then commit_auxiliary_speculative_states, which reaches the target's commit.
- Target:
  - GdpReplayState holds the draft caches and registers the commit with the spec worker. It resets pending where
    requests start and after a one-token step.
  - The caches are keyed by the V2 manager's request index (2 x max_batch + 1 rows, 7.09 MB each at D = 2). That
    bounds them, the memory profile counts them, and they survive V2 suspend and eviction.
  - gdp_replay gained cache_rows (4 cells, 1 guard).
  - MTP off keeps gdp_recurrent (a tie at 24 heads); D >= 3 keeps gdp_verify (faster at T = 4).
- Speed against int-tp1 (same GPU, 2 ABBA rounds):
  - MTP-2 verify step at C = 1 / 16 / 64 / 256: +2.0 / +1.3 / -0.1 / +0.6% (5.219 -> 5.324 ms at C = 1);
  - 8K prompt step: -0.2%; C = 1024 mixed step: -0.1%.
- Memory at max batch 448: KV 45.33 -> 68.32 GiB (+51%), 421 -> ~711 full 8K + 1K requests by pareto-v3-tp1's
  count. D = 3 to 5 are unchanged.
- Top end (16K step, C = 512 / 1024), against a baseline of 7,798 / 7,873 tok/s/GPU (mean of 4 runs, spread
  +-0.8 / +-0.4%):
  - max batch 448: 8,031 / 8,157 tok/s/GPU, +3.0 / +3.6%, tok/s/user -8%;
  - max batch 640: 8,141 / 8,251 tok/s/GPU, +4.4 / +4.8%, tok/s/user -16 / -23%.
- Accuracy at D = 2, against the int-tp1 tree:
  - B(1) passes: 3 of 10 prompts identical and 7 near-tie partings within 1.75 (the baseline: 2 and 8).
  - B(2) passes: AL 2.764 against the reference's 2.7675.
  - GSM8K MTP off / MTP 2: 56.63 / 56.03 (the baseline: 56.63 / 56.63). MTP 2 gives MTP off's answer on 93.9% of
    the questions (the baseline: 94.1%).
- Numerics: an accepted draft's state update is folded in at the request's next step, from fp32 caches, instead of
  being read from a per-position state. The verify-invariance drift bands were re-derived by the file's rule:
  2.25 / 1.16 -> 2.39 / 1.53.
- Under load, every replay-tree run logs one recovered CUDA allocator OOM warning; the baseline logs none. No run
  failed, and a leak check found no leak. The cause is not identified.
- Base: int-tp1
- Full report: `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/nemotron4-mv2-v3-ws/task_results/task_result_side-gdp-replay-tp1.md`

**Goal.** Side task: not merged, its branch kept for a human. Wire `gdp_replay` into tp1, which `int-tp1` left out.
- **Why it was left out:** `gdp_replay` needs an engine change, because the MTP worker's `update_mamba_states` overwrites the state that `gdp_replay` manages (`WS/task_results/task_result_int-tp1.md`).
- **What it would give** (`survey-gdp-recurrent`):
  - verify 1.14–1.53x faster at D = 2, though `int-base` measured it slower than `gdp_verify` + commit at D ≥ 3;
  - and no per-draft intermediate states, ~11 GiB at D = 3 and max batch 448. tp1's top end is bound by `max_batch_size` and KV (`pareto-v3-tp1`: 320 of 320 rows at D = 4 + 16K, KV 86%).
- **Built on `int-tp1`'s tree.**
- **The engine change:** the smallest one that lets a target own its recurrent-state commit, e.g. a target hook that makes the worker skip `update_mamba_states`. Say which other models' paths it touches; it must leave them unchanged.
- **Wire `gdp_replay`** where it wins: D = 2 verify, and decode. Keep `gdp_verify` where that is faster.
- **Measure:**
  - the verify step at C = 1, 64 and 256;
  - the KV budget at D = 2–5;
  - the protocol's top end (C = 512 / 1024) with `max_batch_size` raised to what the freed memory allows.
- **Accuracy:** B(1) and B(2) at the delivered draft lengths, with the gate bands.

**Definition of done.**
- The measurements.
- A `WS/perf_ledger.md` row.
- The result's `summary` gives the change, the speedup and its shapes, and the effect on accuracy, for `UNMERGED_OPTIMIZATIONS.md`.

**C++ changes.** If the tree carries C++ / CUDA changes (`git diff b22283a1b6 -- cpp/`), build it per `WS/test_commands.md` → *C++ build* (`env-cpp`) before any GPU run.

**Output size (read this).** A tool output over 1 MB ends your session: `report-prep`'s attempt 1 died of `CLIJSONDecodeError('Failed to decode JSON: JSON message exceeded maximum buffer size of 1048576 bytes...')` after doing its work and before writing its result. Never print a whole large file or an unbounded diff (`perf_model.md` is 258 KB; server logs and `sweep.json` dumps run to several MB). Use `head`, `tail`, `grep` and `wc`, and keep every command's output under ~100 KB.

**Waiting (read this).** Your session ends the moment you reply without a tool call. Background commands, Monitors and detached pollers do not resume it; the orchestrator then demands your result, and the task fails unverified. So while any GPU job is queued or running, every reply is a foreground tool call: `timeout 540 bash -c 'until [ -f <sentinel> ]; do sleep 30; done'`, repeated until the sentinel exists, then the next step. The pool is full, and a whole-node job can queue for an hour. `op-cells-16k` and `mtp-identity-dep4` attempt 1 ended this way after writing everything and running nothing.

**GPUs.** One GPU, at `perf` priority.

## side-tp1-topend

- Branch: `exp/nemotron4-mv2-v3-unmerged/side-tp1-topend`
- Result: DONE: Side task, nothing to merge. The change is serve configuration only, and I recommend against adopting it.
- Change, tp1 MTP-Eagle with the 16384-token chunked step at kv_cache_config.free_gpu_memory_fraction 0.85:
  max_batch_size (and cuda_graph_config.max_batch_size) 320 -> 328 at D = 4 and 288 -> 296 at D = 5.
- Why those sizes and not the report's 343 / 316: those were what the KV holds at the old batch. Each added row
  also costs ~0.17 GiB (D = 4) / 0.20 GiB (D = 5) of profiling peak (mostly the per-position GDP states the
  hybrid cache allocates per row, plus row-sized workspaces) outside the KV pool, and 27 MiB of GDP slots inside
  it: ~2 full requests of KV a row. The engine's KV lines matched this model to 0.04 GiB at
  every batch I ran.
- Measured, same GPU and session, A/B, benchmark protocol, exact 8192 / 1024 lengths, C = 512 / 1024 / 2048:
  - D = 4: tok/s/GPU +2.20 / +0.74 / +0.01% (8,211 -> 8,212 at C = 2048); tok/s/user +1.49 / +0.23 / -0.77%.
  - D = 5: tok/s/GPU -0.06 / +0.48 / +0.10%; tok/s/user -1.95 / -1.63 / -1.88%.
  - All within the stated ±1% band. Expected +0.3% tok/s/GPU and -2.2% tok/s/user.
- Queueing (iteration stats, C = 1024): every row is used and the KV becomes the bound.
  - B = 328 runs 328 of 328 rows with 193 queued, 0 paused, KV 97.0% used.
  - B = 296 runs 296 of 296 with 152 queued, 0 paused, KV 95.7%.
  - The report's B = 343 is capped by the KV at 300 running rows: -0.7% tok/s/GPU, +8% tok/s/user against 328.
- Why nothing moves: at the prompt-bound end every step must admit rows / ~220 prompts, so the step time grows
  with the rows (T ~ 8.6 + 0.51 x rows ms). More rows trade tok/s/user for tok/s/GPU almost one for one.
- Accuracy: no effect. The forward is unchanged; the only new CUDA-graph sizes (328, 296, 343) pass
  thop_attention's verify within its band (22 of 22 cases), and gdp_verify / causal_conv1d_verify have certified
  cells at graph batches 256 and 448 for both draft lengths.
- Recommendation: keep 320 / 288. A ledger row was added.
- Base: int-tp1
- Full report: `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/nemotron4-mv2-v3-ws/task_results/task_result_side-tp1-topend.md`

**Goal.** Side task: not merged, its branch kept for a human. tp1's throughput end, the top of the combined frontier, binds on `max_batch_size`. D = 4 with the 16384-token step runs 320 of 320 rows with 143–145 requests queued and 14% of its KV free (`pareto-v3-tp1`'s iteration-stats companion). A larger max batch is untested.
- **Work on `int-tp1`'s tree.** Raise `max_batch_size` for D = 4 and D = 5 with the 16K step to what the KV holds at fraction 0.85, the report's recommendation. Extend the CUDA-graph sizes, and the cells of every entry the larger batch reaches, as needed.
- **Measure** C = 512, 1024 and 2048 under the protocol: tok/s/GPU, tok/s/user, and the queueing.
- **Expected,** stated first: the steady part of C = 1024 ran at 8,416 tok/s/GPU against the point's 8,213, so the ceiling is a few percent.

**Definition of done.**
- The measured points.
- A `WS/perf_ledger.md` row.
- The result's `summary` gives the change, the gain and its shapes, and the effect on accuracy (none expected), for `UNMERGED_OPTIMIZATIONS.md`.

**C++ changes.** If the tree carries C++ / CUDA changes (`git diff b22283a1b6 -- cpp/`), build it per `WS/test_commands.md` → *C++ build* (`env-cpp`) before any GPU run.

**Output size (read this).** A tool output over 1 MB ends your session: `report-prep`'s attempt 1 died of `CLIJSONDecodeError('Failed to decode JSON: JSON message exceeded maximum buffer size of 1048576 bytes...')` after doing its work and before writing its result. Never print a whole large file or an unbounded diff (`perf_model.md` is 258 KB; server logs and `sweep.json` dumps run to several MB). Use `head`, `tail`, `grep` and `wc`, and keep every command's output under ~100 KB.

**Waiting (read this).** Your session ends the moment you reply without a tool call. Background commands, Monitors and detached pollers do not resume it; the orchestrator then demands your result, and the task fails unverified. So while any GPU job is queued or running, every reply is a foreground tool call: `timeout 540 bash -c 'until [ -f <sentinel> ]; do sleep 30; done'`, repeated until the sentinel exists, then the next step. The pool is full, and a whole-node job can queue for an hour. `op-cells-16k` and `mtp-identity-dep4` attempt 1 ended this way after writing everything and running nothing.

**GPUs.** One GPU, at `perf` priority.

## side-moe-persistent

- Branch: `exp/nemotron4-mv2-v3-unmerged/side-moe-persistent`
- Result: DONE: Side task, not for merge as is. Built and measured the one-queue persistent kernel; it is not wired into any target.
What changed: new catalog entry latent_moe_experts_persistent (MV2/catalog/moe/, Python + Gluon, with a GPU test).
It is a drop-in for latent_moe_experts_fused_act (fa). For prompt-sized batches, both routed-expert GEMMs run in
one persistent, warp-specialized Gluon kernel:
- one CTA per SM takes up+act and down tiles from one atomic queue;
- a down tile starts once its entry's 16 up slices are stored (per-entry release / acquire counters);
- optionally, the intermediate h lives in a caller-owned ring that the caller marks L2-persisting.
Accuracy: unchanged. The output equals latent_moe_experts' (and fa's) bit for bit on every measured case
(tp1, dep4 windows, tp4) and every test cell.
Speed per call against fa (same GPU, CUDA-graph replay, real routing):
- h full size, no L2 configuration:
  - tp1: 8192 tokens -1.5% (1256 -> 1238 us), 16384 +1.4%;
  - dep4 slowest window: -1.9%;
  - tp4: 8192 +6.6%, 16384 +14.9%.
- h in a 96-slot ring with a 24 MB persisting set-aside, tp4: 8192 -6.7% (416 -> 388 us), 16384 -3.0%. fa measured
  with that set-aside held is unchanged (+0.2% at most). This is the one clean win.
- h in a 48-slot ring with a 48 MB set-aside, tp1: 8192 -6.2% (1256 -> 1179 us), 16384 -1.8%. But that set-aside
  slows kernels that do not use it: fa itself +2.1% / +4.0% while it is held. Rings under 48 slots stall at tp1.
- Kernel only, tp1 8192: fa's two GEMMs 1154 us; fused, h full size 1128 us (-2.2%); fused, 64-slot persisting
  ring 1063 us (-7.9%).
The -14% bound was not reached. h does not stay in L2 on its own, and an evict-first hint on the weight loads is
10-15% slower (the weights must stay in L2 between the entries of one expert). 16K is compute-bound, where the
Gluon tile loop trails fa's kernels.
- Base: int-tp1
- Full report: `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/nemotron4-mv2-v3-ws/task_results/task_result_side-moe-persistent.md`

**Goal.** Side task: not merged, its branch kept for a human. A long shot at the largest lever left at tp1's throughput end, which is bound by prompt admission (`pareto-v3-tp1`, `pareto-v3-dep4`). The lever is one persistent, warp-specialized kernel that runs both routed-expert GEMMs from one queue on every SM: `perf-moe-streamed-mlp`'s conclusion, with a bound of about -14% per MoE call at tp1 8K.
- **Why the earlier design lost** (`WS/task_results/task_result_perf-moe-streamed-mlp.md`). The two-stream version gave no gain: half the SMs make each GEMM about 1.5x slower, plus handoff bubbles. One queue on every SM avoids the split.
- **Built on `int-tp1`'s tree,** whose prefill MoE is `latent_moe_experts_fused_act`.
- **Expected** (stated first): about -14% per call, roughly -5 ms per 8K prompt step, roughly +4–5% tok/s/GPU at tp1's top end. The risk is high, and it is a multi-day kernel at full scope.
- **Time box:** about 6 h. Deliver the kernel if it measures faster, or the design and the measured partial results.
- **Accuracy:** keep `latent_moe_experts`' bits if possible; otherwise use bands against its reference.

**Definition of done.**
- The measured result per call against `latent_moe_experts_fused_act` at 8K and 16K.
- A `WS/perf_ledger.md` row.
- The result's `summary` gives the change, the speedup and its shapes, and the effect on accuracy, for `UNMERGED_OPTIMIZATIONS.md`.

**C++ changes.** If the tree carries C++ / CUDA changes (`git diff b22283a1b6 -- cpp/`), build it per `WS/test_commands.md` → *C++ build* (`env-cpp`) before any GPU run.

**Output size (read this).** A tool output over 1 MB ends your session: `report-prep`'s attempt 1 died of `CLIJSONDecodeError('Failed to decode JSON: JSON message exceeded maximum buffer size of 1048576 bytes...')` after doing its work and before writing its result. Never print a whole large file or an unbounded diff (`perf_model.md` is 258 KB; server logs and `sweep.json` dumps run to several MB). Use `head`, `tail`, `grep` and `wc`, and keep every command's output under ~100 KB.

**Waiting (read this).** Your session ends the moment you reply without a tool call. Background commands, Monitors and detached pollers do not resume it; the orchestrator then demands your result, and the task fails unverified. So while any GPU job is queued or running, every reply is a foreground tool call: `timeout 540 bash -c 'until [ -f <sentinel> ]; do sleep 30; done'`, repeated until the sentinel exists, then the next step. The pool is full, and a whole-node job can queue for an hour. `op-cells-16k` and `mtp-identity-dep4` attempt 1 ended this way after writing everything and running nothing.

**GPUs.** One GPU, at `perf` priority.
