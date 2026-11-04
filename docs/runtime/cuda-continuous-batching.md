# Flash-Next CUDA continuous batching

Enable an explicit request capacity with `mfq serve --max-requests-per-runtime N`
or native `mfq-runtime --continuous-batching N`. Flash-Next supports capacities
from 1 to the configured sequence limit on one CUDA device. The limit defaults
to 32 and can be reduced with `MFQ_CONTINUOUS_BATCH_MAX_SEQUENCES`. Values above
32 are rejected in both the native worker and Python server. GDN independent
batch preparation currently accepts at most 32 rows. The managed server defaults to eight requested
rows and adopts the worker's reported `batch_capacity`; models without
per-request batch state keep the single-request path and log a warning instead
of failing the load. See [multi-request concurrency](concurrency.md).

Each request owns its GDN recurrent state, PLE convolution and ngram history,
QSA caches, token counts and logical position. Pending prefills run in bounded
chunks. Eligible decode requests share GR, routing and FFN computation, including
the expert transfer and decode pass. Paused requests do not advance their state.
Cancellation releases only the cancelled request's state.
Eligible rows wait for their peers to schedule the next decode, so requests
with different prefill lengths form a batch instead of alternating single-row
decodes. A request paused by output backpressure does not hold up other rows.

The native CUDA decode window caches CUDA Graphs in row buckets of 1, 2, 4, 8,
16 and 32, capped by the configured request capacity. It selects the smallest
bucket that fits the eligible requests. The Qwen and Flash-Next adapters share
this policy, static row metadata and device state-pointer tables. Arrivals,
departures, pauses and row reordering update these inputs before replay. A new
row bucket, length bucket or cache capacity can require capture.
Padding rows cannot commit request KV, recurrent or PLE state, ngram history,
token counts or expert routes; their outputs are discarded.
With dynamic KV allocation, extra request contexts reserve stable virtual
addresses and map physical QSA pages only for admitted prefill/decode tokens.
The initial exclusive context retains its existing warm mapping.
Managed preallocated KV retains its configured full-context allocation policy;
admission charges that physical allocation before starting another request.
Managed KV bytes are counted through `KvMemoryManager`; request-owned recurrent
state and other KV buffers are included in the runtime's request KV metrics.

Before starting a prefill, the shared engine scheduler reserves the uncached
prompt and expected remaining decode for each admitted request, including paused
and partially prefilled requests. Expected decode is
`ceil(min(max_new_tokens - produced, decode_token_clip) * new_token_ratio)`.
CUDA converts these tokens to KV bytes using cache geometry and allocation
rounding, including VMM mapping granularity and full-context indexer storage.
Preallocated storage is charged up front; already allocated decode capacity
requires no additional reservation. The memory gate uses actual available bytes
plus reclaimable prefix hot-cache and optional expert workspace bytes; paged KV
also includes allocated free pages. Devices have separate budgets. A 1 GiB device
workspace allowance plus expert transfer workspace stays outside the KV budget.
Cache owners release hot entries before allocating the admitted prefill.

Requests that do not fit stay queued and are retried on subsequent steps.
Cancellation still releases a queued or partially prefilled request. With no
other admitted request, a prompt plus one decode token may start even when its
expected future decode does not fit, so a long output limit cannot prevent the
only request from making progress. The complete uncached prompt must still fit.
An optional `MFQ_CONTINUOUS_BATCH_KV_BUDGET_BYTES` caps the batch's KV budget
(default zero: use actual device capacity); it never increases physical capacity.

Flash-Next checks growth before decode and retracts requests under pressure.
Shortest generation is retracted first, with longest prompt breaking ties. At
least one running request and one eligible decode row are retained. Retraction
retires graphs and frees the request's KV directly, without inserting it into
the prefix cache. The request returns to the prefill queue and reconstructs KV
from its prompt and accepted tokens, excluding the pending decode input. Replay
is chunked and publishes no tokens; sampler/RNG, penalties, accepted output and
generation limits survive. If the last request cannot grow, the existing
context-capacity error remains. Qwen continuous batching uses the admission gate
but retains its existing decode-capacity failure behavior; it does not retract
shared stable-slot state. Unexpected CUDA failures still fail the backend.
MTP verification reserves its complete padded token window before target execution.
Retraction retains the depth policy and accepted-token history; replay rebuilds
the predictor context before speculation resumes.
This does not resize the resident expert cache or automatically enable KV offload.

The ratio follows SGLang's scheduler policy: initial
`min(0.7 * schedule_conservativeness, 1)`, minimum `0.14 * initial`, and linear
decay `(initial - minimum) / 600` after every successful decode step. Retraction
recomputes it from the surviving requests as
`min((sum(produced) + 20 * running_count) / (sum(max_new_tokens) + 1), 1)`.
The next decode decay applies the minimum; an idle batch resets to the initial
value. Environment overrides are `MFQ_INIT_NEW_TOKEN_RATIO`,
`MFQ_MIN_NEW_TOKEN_RATIO_FACTOR`, `MFQ_SCHEDULE_CONSERVATIVENESS`,
`MFQ_NEW_TOKEN_RATIO_DECAY_STEPS`, `MFQ_RETRACT_DECODE_STEPS` and
`MFQ_ADMISSION_DECODE_TOKEN_CLIP` (default 4096). Ratios/factors must be in
`(0, 1]`, conservativeness positive, and step counts/clip positive integers.
These values also have fields in `ContinuousBatchConfig::admission`.

With `MFQ_CONTINUOUS_BATCH_MTP=1`, CUDA continuous batching runs greedy MTP
requests inside the batch with the Qwen and Flash-Next predictors, including
token penalties and grammar constraints. Each request owns independent
predictor history and a depth controller. Draft work groups
equal history widths, then batches the rows still drafting at each chain
position. Target verification runs once with width `1 + max(K_i)`; shorter
chains use padding. Each row commits only the prefix emitted after EOS, stop
and output-limit handling. Rejected suffixes restore GDN/PLE checkpoints and
truncate QSA history. Paused rows keep their pending transaction; cancellation
releases it without committing a draft suffix. Greedy sampling consumes no RNG
draws. Parked rows accumulate teacher-forced history and flush it at 16 pairs
or before re-entering speculation.

Without that setting, MTP requests stay on the exclusive single-request path,
as before. The batched path is opt-in because this first stage uses eager
draft and verify execution, while the exclusive path replays CUDA Graphs.
It reports
`continuous_batching_mtp_supported=1`,
`continuous_batching_mtp_stochastic_supported=0`, and
`continuous_batching_mtp_graph_supported=0` when a supported predictor is loaded.
Ordinary decode retains the row-bucket graphs described above. MTP graph replay
and row-bucket × verify-width graph keys remain unimplemented. Stochastic MTP
requests remain exclusive and use the existing single-request implementation;
they are not silently converted to ordinary decoding. Batched requests do not
use persistent prefix reuse. Explicit session IDs and multimodal requests
remain exclusive. The default single-request path retains MTP and prefix
caching. Configured QSA KV offload uses the eager batch path.
Cached Qwen MoE retains its eager fallback. This graph cache is specific to
native CUDA. HIP reports `continuous_batching_mtp_supported=0` and keeps MTP
requests exclusive. Metal reports its existing `supports_batch_mtp()` result
(a single predictor layer without MTP LoRA); unsupported MTP requests remain
exclusive. Neither backend uses the CUDA uniform-width implementation.
Entering an exclusive request retires the batch graph cache before its decode
workspace is borrowed.

For a synthetic fixture containing predictor weights, set `MFQ_BATCH_MTP_CHECK=1`
with `--check-continuous-batching` to compare batched MTP against ordinary
single-request greedy decoding. `MFQ_BATCH_MTP_BENCH=1` also measures 1, 4, 8,
16 and 32 requests with MTP off/on. The benchmark uses eager execution for both,
17-token prompts, 48 generated tokens and seed `919 + row`; loading and serial
oracles are outside the timer. `stream_tok_s` averages the per-request rates
after its first token; `total_tok_s` includes prefill and draining the batch.
It reports accepted/drafted counts with each sample. The random small fixture's
acceptance rate is not representative of trained predictor weights.

Run the state check with the native diagnostics executable:

```shell
mfq-diagnostics --model model.mfq --ctx-size 512 --check-continuous-batching
mfq-diagnostics --model model.mfq --ctx-size 512 --check-continuous-batching --continuous-batching-check-requests 32
```

The check compares outputs against serial generation, exercises paused rows,
decode cancellation, partial-prefill cancellation and subsequent request reuse.
The larger check holds all requests until their prefills finish, then decodes
them together across the 256-token cache-capacity boundary. Small native model
fixtures include QSA, GDN, PLE and routed experts. The shared execution test also
injects prefill and decode-capacity failures to verify surviving requests, and
checks staggered request scheduling. The stdio chat-client integration test
compares simultaneous generations with serial output and verifies that the
decoder actually processes multiple tokens per batch.

The diagnostics also exercise arrivals, departures, paused rows and reordered
rows within one bucket, comparing generated tokens with serial generation and
checking that inactive recurrent state stays byte-identical. The existing
`continuous_batching_cuda_graph_captures` and
`continuous_batching_cuda_graph_replays` counters remain available.
`continuous_batching_cuda_graph_membership_captures` counts captures caused only
by membership changes and remains zero. `continuous_batching_cuda_graph_bucket_reuses`
counts subsequent replays of an existing bucket, including changed members or row order.
`continuous_batching_cuda_graph_capture_ms` accumulates graph warmup and capture
time, including instantiation. `continuous_batching_admission_deferred` counts
candidate memory-gate deferrals (a request may be counted on multiple steps),
`continuous_batching_retractions` counts requests returned for KV replay, and
`continuous_batching_new_token_ratio` reports the current estimate. These are
reported through the existing runtime metrics snapshot.

## Allocator and cache growth

CUDA Graph scratch uses a separate asynchronous device pool from resident
weights and reusable quantization workspaces. All row buckets borrow the same
pool and execute serially, allowing their temporary allocations to reuse its
scratch. Request GDN state is allocated on first use in a resident pool.
Retiring the bucket cache trims its unused pages; pool telemetry includes every
live and retired pool. Pool handles are reclaimed when their remaining tensors
are released.

QSA keeps one KV layout while a graph is active. It releases the previous
sequence layout after preparing graph KV, and reconstructs that layout only
when leaving the graph. Batches above 16 requests grow graph capacity in
128-token increments after their initial allocation; smaller batches retain
the previous doubling policy. The configured context limit remains a ceiling,
with no full-context allocation at admission.

Eligible GDN layers process independent requests in one decode core launch.
A small address table points to each request's existing convolution, recurrent
state and counters; recurrence stays in those allocations. One subsequent
kernel commits convolution rows after all readers finish. Membership changes
update the table before replay; null entries skip recurrent and convolution
commits for inactive rows. The fused path keeps recurrence in request allocations.
Other recurrent paths use shared staging slots for indirect loads and commits.
`MFQ_GDN_INDEPENDENT_BATCH=0` restores the per-request core calls.
This dispatch requires batched attention projections and supported fused GDN
geometry. Unsupported layers retain their per-request path.
Request decode counters are prepared after prefill, before graph recording,
and are included in the growth reservation. `MFQ_TRACE_GDN_BATCH=1` reports
state eligibility, projection dtypes and actual batch core selection.

QSA decode also dispatches independent requests together. Shared operators use
the same device row table for KV, raw index and pooled index addresses and
capacities. Normalization, RoPE, cache updates, pooling, scores, selection and
attention launch across rows; attention reads request KV in place. Cache stores
finish before attention launches. Pooling reads the current index directly from
projection input. Disabled rows cannot commit any cache state, even when their
table entries contain valid addresses. Membership changes and cache growth
update the table before replay. F16 attention retains its Tensor Core path at
head width 256, and TurboQuant attention reads packed request KV directly.
`MFQ_QSA_INDEPENDENT_BATCH=0` restores the previous per-request graph core for
comparisons. Configured KV offload retains the existing eager path.
Exclusive GPU/RAM KV tiers also use the independent core. The row table resolves
hot slots and cold positions with each tier's physical capacity; F16/MMA and
packed TurboQuant attention read those allocations directly. A single KV
update launch handles all rows, including eviction before publishing mappings.
`MFQ_TRACE_QSA_BATCH=1` reports independent core recording. These operators are
native CUDA; Metal and HIP keep their existing decode paths.

`mfq-window-attention-test --qsa-independent-batch-check` checks heterogeneous
contexts, paused and cancelled requests, row reuse, cache growth and the
dense-to-sparse boundary at 2, 8, 16 and 32 rows, with contiguous and exclusive
GPU/RAM KV. The corresponding
`--qsa-independent-batch-bench` compares core kernel counts and graph replay
times at 1, 4, 8, 16 and 32 rows. Set `MFQ_QSA_BATCH_TIERED=1` for exclusive
GPU/RAM KV with 16 hot cells per request; `MFQ_KV_TURBOQUANT_BITS` selects the
cache format for both checks and benchmarks. With a synthetic model,
`MFQ_QSA_BATCH_BENCH=1 mfq-diagnostics --model model.mfq --ctx-size 512 --check-continuous-batching`
also measures complete decode steps at these batch sizes and verifies their
tokens against serial generation.

Unexpected quota changes after the admission gate can still reject a prefill
without poisoning the backend. Decode growth failures before numerical execution
can retract a row; a single remaining row retains the context-capacity error. Other CUDA failures retain their existing
fatal-error behavior. Native tests inject failures after transfer submission
and during graph recording, then check subsequent execution and state.

For 17–32 rows, large NINT vocabulary projections read packed weights through
the online Tensor Core path. Other projections keep their existing GEMM path.
`MFQ_NINT_COMPRESSED_LARGE_BATCH=0` disables this dispatch; `head` selects the
default, and `1` enables it for other eligible projections as an experiment.
Decoded expert weights are not kept as persistent expanded allocations.

## Local validation, 2026-10-12

S2-L on an RTX 3090 Ti, 4096 context limit, 2048-token prefill chunk,
32-request capacity, 10.66 GiB expert residency and 22.5 GiB process VRAM limit.
The 32-request case follows two 1024-token MTP requests in the same process,
generates 256 tokens per request, and crosses the 256-token KV boundary.

| Case | Completed tokens | Total throughput | Decode graph throughput |
| --- | ---: | ---: | ---: |
| 32 requests, 256 tokens each | 8192 | 87.55 tok/s | 100.01 tok/s |
| 32 distinct prompts, 24 tokens each | 768 | 43.06 tok/s | 96.24 tok/s |

Both cases complete without errors and match their serial text references.
Total throughput includes prefill and graph capture; graph throughput measures
CUDA window replay only. These are single-run observations, not an isolated
A/B speedup claim. The distinct-prompt case uses a separate serial reference
for each request. Additional validation exercises 16 staggered arrivals and
checks ordinary decoding and MTP before and after large batches.
After both 32-request cases, 1024-token ordinary decoding measured 50.30 tok/s
and MTP measured 74.05 tok/s from streamed output after the first token.
Both continuations match the serial text reference. Cold MTP is slower;
these measurements describe the warmed process and do not establish a speedup.

With indirect GDN enabled under the same configuration, the 32 × 256 case
completed all 8192 tokens at 89.20 tok/s total and 101.71 tok/s graph throughput.
The distinct-prompt case measured 98.83 tok/s graph throughput; 16 staggered
requests completed all 2048 tokens at 111.94 tok/s graph throughput. Ordinary
decoding and MTP afterwards measured 50.40 and 73.27 tok/s from streamed output.
All outputs match their serial references. Independent-run variations should
not be treated as isolated speedup measurements.

The S2-L graph shrinks from 13,438 to 10,054 nodes, including a reduction from
3,200 to 896 copy nodes. A same-process GDN primitive comparison at 32 rows
measured 442.03 µs for independent launches and 289.92 µs for indirect batching.
Output, convolution and recurrence are byte-exact. Tests exercise changing
inputs and state address order at 2, 4, 6, 8, 16 and 32 rows with 4, 2 and 1
column tiles. The model fixture also asserts that 32 rows enter the new core.

Short GDN verification sequences use two columns at length 2 and four at
lengths 3–8 for eligible D128 layers. `MFQ_GDN_SHORT_COLUMNS=0` restores the
previous short-sequence dispatch; an explicit `MFQ_GDN_PREFILL_COLUMNS` value
retains priority. Single-token decode and long prefill dispatch are unchanged.
Same-process primitive measurements reduced length-2 recurrence from 7.17 to
6.51 µs and length-6 recurrence from 14.68 to 13.15 µs. Validation covers 360
exact multi-column cases, changing graph inputs, recurrence state, prefill to
decode transitions and the full native window test.

With this dispatch, two warm 1024-token S2-L MTP requests measured 73.95 and
74.57 tok/s from streamed output; ordinary decode measured 50.19 tok/s.
A subsequent 16 × 128 request batch completed all 2048 tokens at 93.18 tok/s
total and 119.54 tok/s graph throughput. All outputs match the reference.
These observations do not establish an isolated whole-model speedup.

Compact E8 projections retain the immutable JSC state-to-bank map in four
previously unused descriptor bytes. The map is prepared on the CPU while
loading weights and follows each shared codebook into expert slices and
transfer descriptors. This avoids one dependent metadata lookup per group;
weights, codebook storage and descriptor size stay unchanged. Zero maps retain
the previous lookup path, including valid single-bank codebooks.
`MFQ_MFE_NVQ_BANK_MAP=0` selects the previous path when constructing FFN views.

Real S2-L FFN graph checks cover 252 changing-input cases at 1, 2, 3, 4, 5, 6
and 8 tokens, shared experts, repeated routes and invalid routes. Output and
active hidden values match byte for byte; the fixture verifies the actual
compact descriptors contain their codebook maps. Six native metadata fixtures
also verify every state and expert slice, and eight NVQ request-batching cases
pass. Same-process complete FFN timings show small gains or variations around
zero, usually below 1%; these are not a whole-model speedup claim.
The complete-model regression finishes 32 × 256 tokens without errors at
101.64 tok/s graph throughput. Subsequent ordinary decode measures 50.00
tok/s and warm MTP measures 75.32 tok/s from streamed output; the earlier two
warm MTP runs measure 73.65 and 73.94 tok/s. All continuations match the
reference, including the separate serial oracles for 32 distinct prompts.

Independent request decoding now gives eight-row FFN tiles an eight-row
expert reuse hint. Previously the tile size selected the two-row speculative
heuristic even when the original request batch contained 16 or 32 rows.
The hint also enables grouping for pure NINT tiles. Speculative verification
at 2–6 rows retains its existing dispatch, and an explicit
`MFQ_MFE_TOKEN_GROUP_CAPACITY=2` overrides the hint for comparisons.

An actual S2-L complete FFN comparison at eight rows measures 423.77 µs with
two-row reuse and 350.61 µs with eight-row reuse, including shared experts.
Changing graph inputs and routes match byte for byte in 36 cases. The full
native grouping fixture passes 162 cases across NINT, NVQ, shared experts,
repeated routes and distinct routes; 24 request-state cases also pass.
The S2-L 32 × 256 regression completes all 8192 tokens at 90.92 tok/s total
and 103.92 tok/s graph throughput, with actual eight-row dispatch confirmed
by the runtime trace. Ordinary decode afterwards measures 50.27 tok/s and
MTP 73.57 tok/s from streamed output. All references match, including 32
distinct prompts. Whole-model measurements remain independent-run observations.

The same compact E8 bank map now serves the grouped token decoder, including
short MTP verification and eight-row independent FFN tiles. Real-weight checks
cover 252 changing-input cases at 1–8 rows and another 36 capacity-eight cases;
outputs and active hidden values match byte for byte. Complete FFN timings
remain close to the original path: length three with shared experts measures
205.16 versus 204.60 µs, and capacity eight with shared experts 349.79 versus
349.84 µs. These small results do not establish a whole-model gain.

A scope-three profile of 32 × 24 S2-L generation measured 315.79 ms of GPU
time per replay. Attention-end to FFN-pre accounted for 72.00 ms; the previous
FFN's transfer-end to next attention-start accounted for 69.08 ms. These
intervals include GR and boundary work and are not pure kernel timings.
Resident expert calculation accounted for 54.70 ms and attention cores
39.28 ms. All 768 generated tokens matched the reference. Profiling adds
events, and this run also enabled host serving traces; its throughput is not
the production benchmark. `bench/analyze_decode_window_profile.py` selects
the largest task-count window by default and keeps tile boundaries separate.

The GR two-stage dispatcher previously accepted at most eight rows. Larger
independent request batches selected the separate composition; at exactly
32 rows that composition also selected the GR prefill projection kernel.
Continuous decode now accepts 9–32 rows and shares each aligned Up decode
between two rows. Ordinary single-row decode, short verification and prefill
dispatch retain their existing selection. `MFQ_GR_CONTINUOUS_TWO_STAGE=0`
restores the previous continuous dispatcher; `MFQ_TRACE_GR_DISPATCH=1`
reports the selected row counts.

The real-weight GR probe rotates twelve attention/FFN banks from six S2-L
layers. Half/Float inputs, pre/after composition and three changing graph
inputs at 9/16/17/32 rows pass 48 cases and 1728 tensor comparisons. At 32
rows the correctness reference explicitly disables the prefill projection,
preserving its ordered FP32 alternative. The timing baseline retains the
old prefill selection. FP32 GR at 16 rows measures 315.18 → 244.05 µs before
attention and 316.92 → 244.88 µs after attention. At 32 rows it measures
1462.19 → 449.37 and 1463.42 → 450.50 µs respectively. These complete GR
measurements are not whole-model throughput claims.

Reproduce with `run_gr_batch_probe.py check` and `run_gr_batch_probe.py bench`
after building `mfq-cuda-gr-batch-probe`. Its paired graphs share one prepared
pool across fixture cases, following the native projection fixture workflow.
An earlier variant which retired separate fixture pools between cases crashed
during the reference graph's replay; memcheck reported no GPU errors but did
not finish successfully. That failed run is not accepted as validation.
The runtime allocator has not been changed by this optimization; full-model
graph transitions and KV growth are covered by the following regression.

After three MTP requests, the S2-L 32 × 256 run completes all 8192 tokens at
129.40 tok/s total and 156.56 tok/s graph throughput. The previous uploaded
checkpoint measured 90.92 and 103.92 respectively; these are independent-run
observations. Runtime tracing confirms `gr_two_stage rows=32 continuous=1`,
and the batch graph decreases from 11590 to 10918 nodes. A subsequent
16 × 128 run completes all 2048 tokens at 129.83 tok/s graph throughput.
Ordinary 1024-token generation after these batches measures 50.19 tok/s,
and MTP 73.61 tok/s, from streamed output. All references match, including
32 distinct prompts checked against their individual serial responses.
That heterogeneous batch measures 140.61 tok/s graph throughput. The earlier
two warm 512-token MTP requests measure 70.46 and 73.14 tok/s.

The 32-row test crosses its initial KV capacity without changing the 22.5 GiB
process limit or 10.66 GiB resident expert budget. It reports no errors;
request KV is released afterwards and free VRAM is 6.54 GB. Native NVQ
request-state cases at 2/4/6/8/9/16/17/32 rows pass, along with the 162-case
MFE grouping suite, full native FP32 projection/GR suite and nine batching
source checks. The production graph allocator remains unchanged.
