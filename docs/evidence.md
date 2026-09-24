# FORGE: the long-form record

Every table here is checked against its artifact by `scripts/check_readme_claims.py`, which fails the build when a published figure and its committed JSON disagree. The short version, with the charts, is [docs/index.html](https://akilalours.github.io/Forge_Panagram/) and the [README](../README.md).

---

## 🎯 The Question

Detectors are trained on synthetic text somebody had to generate. Almost everyone generates
it the easy way: prompt a model for essays on assorted topics and label them AI. FORGE asks
whether generating synthetic text that **mirrors real human documents**, matched on topic,
length, register and structure, produces a detector that survives contact with generators it
has never seen.

| | Arm A · control | Arm B · treatment |
|---|---|---|
| Human half | FineWeb, identical | FineWeb, identical |
| Synthetic half | **randomly prompted** generations | **matched mirrors** of the human documents |
| Backbone | DeBERTa-v3-base | DeBERTa-v3-base |
| Precision, schedule, seed, budget | identical | identical |

One variable. That is the whole design, and it is the reason the numbers below mean
anything.

---

## 📊 What Actually Happened

### In distribution, both arms saturate and neither number means much

| Metric | Arm A · random | Arm B · mirrors |
|---|---|---|
| AUROC | 0.999989 | 0.999895 |
| FNR at the 0.1% FPR budget | 0.183% | 6.061% |
| Expected calibration error | 0.002552 | 0.004710 |
| Deployed threshold | 0.996586 | 0.998252 |
| Realised FPR | 0.0502% | 0.0502% |

**These two columns are not a comparison.** Each arm is validated against its own AI
source, so arm A is scored on random-synthetic text and arm B on matched mirrors, which are
different test sets. Arm B's higher false-negative rate says mirror text is harder to
detect than random synthetic text, which is what "matched mirrors are harder negatives"
predicts, and it says nothing about which detector is better. The comparison happens in the
next section, on data neither arm has seen.

Both arms clear 0.9998 AUROC on their own held-out split. Human web text against four
open-weight models between 1.7B and 3.8B is a wide gap, and a near-perfect score on it is
close to the expected result rather than evidence of a strong detector.

### Out of distribution, the mirror arm wins on every benchmark

Every arm evaluated on 4,000 documents per benchmark, 2,000 human and 2,000 AI, at the
threshold each arm actually deploys.

| Benchmark | AUROC · A | AUROC · B | Miss rate · A | Miss rate · B | ECE · A | ECE · B |
|---|---|---|---|---|---|---|
| **HC3** | 0.796 | **0.950** | 87.5% | **62.2%** | 0.395 | **0.153** |
| **RAID** | 0.719 | **0.787** | 99.8% | **79.5%** | 0.497 | **0.292** |
| **MAGE** | 0.597 | **0.612** | 97.0% | **89.1%** | 0.463 | **0.387** |

Six comparisons, six in the same direction. Consistency across three benchmarks with
different generators and different domains carries more weight than any single figure.

### Is the gap real, or is it sampling noise

Two tests, because they answer different questions. A paired bootstrap over 10,000
resamples asks whether the *ranking* gap survives resampling; both arms are scored on the
same resampled documents because their errors are correlated.

| Benchmark | AUROC gap | 95% CI | resamples where the gap reverses |
|---|---|---|---|
| **HC3** | +0.1541 | 0.1422 to 0.1660 | **0 of 10,000** |
| **RAID** | +0.0685 | 0.0582 to 0.0790 | **0 of 10,000** |
| **MAGE** | +0.0158 | 0.0035 to 0.0279 | 65 of 10,000 |

AUROC compares rankings, but a detector is deployed at an operating point. So each arm's
threshold was re-fit to spend the **same** human false-positive budget and McNemar's test
run on the discordant pairs.

| Benchmark | B catches, A misses | A catches, B misses | χ² | p |
|---|---|---|---|---|
| **HC3** | **527** | 122 | 251.49 | 7.3 × 10⁻⁶¹ |
| **RAID** | **372** | 34 | 279.73 | 5.3 × 10⁻⁷³ |
| **MAGE** | 64 | 46 | 2.63 | **0.105** |

**The finding, stated honestly.** On HC3 and RAID the mirror arm is better by every measure
and both tests agree decisively. On RAID, the benchmark where the two arms used to be
indistinguishable, the sign of the AUROC gap now reverses in **0.0% of 10,000 paired
bootstrap resamples**, and where the arms disagree at a matched budget the mirror arm is
right 11 times as often. On HC3 it is right 4.3 times as often. Matched generation
transfers.

On MAGE the two tests disagree, and the disagreement is the honest answer. The bootstrap
says the ranking gap is real but 0.016. McNemar says it does not reach the decisions,
p = 0.105. Both arms sit near chance at 0.60. The sentence is that both arms fail on MAGE
and the mirror arm fails imperceptibly less.

### The limitation, in the same breath

> **Neither arm is deployable out of distribution, and the winning arm is not close.**
> Against unseen generators the mirror arm still misses **62% to 89%** of AI text at its
> deployed threshold, and the control arm misses up to **99.8%**. Calibration collapses from
> ECE 0.003 in distribution to **0.15 – 0.46** outside it.
>
> Read those two rows together and they say something more specific than "it gets worse".
> An AUROC of 0.950 on HC3 means the model can still *rank* those documents well. A 62%
> miss rate at the deployed threshold means the operating point calibrated in distribution
> is worthless outside it. **Ranking partially transfers. The threshold does not transfer at
> all.** Any deployment across a domain shift has to re-fit its threshold on the new domain,
> and this system gives you no way to know when it has crossed one.
>
> A confident score from this system is not evidence of a confident model. The live demo
> will show you this if you paste in ChatGPT output, and the page says so rather than hiding
> it.

### Two conditions that bound every number above

> **The generator roster is small.** Four instruction-tuned models between 1.7B and 3.8B,
> chosen so Phase 2 fits in 1.5 GPU-hours instead of 22. Smaller models produce more
> detectable text, so absolute FPR and FNR are optimistic relative to a 7B-to-70B roster.
> The comparison *between* arms is unaffected, because both arms draw the same roster.
>
> **The AI half ran longer than the human half before length matching.** Measured on the
> v0.2-min corpus: a median of 296 words against the human corpus's 257. Both arms pass
> through the identical validator, so the arm comparison was never at risk, but an unmatched
> AI pool would let the detector learn length instead of learning AI. `ai_reference: human`
> resamples each arm's AI pool so its word-count histogram tracks the human corpus before
> training.

### Both training runs are tracked, and the link is backed by an artifact

| Arm | W&B run | AUROC | run record |
|---|---|---|---|
| A · random | [`ms2oreiz`](https://wandb.ai/akilalourdes-student/forge/runs/ms2oreiz) | 0.999989 | `reports/experiments/indist_baseline.json` |
| B · mirrors | [`97c519m6`](https://wandb.ai/akilalourdes-student/forge/runs/97c519m6) | 0.999895 | `reports/experiments/indist_mirror.json` |

Pasting a dashboard link into a README is a claim anybody can make. Each run record here
carries its own `tracking.run_url`, alongside the `code_commit` and `dataset_version` that
produced it, and `test_the_readme_only_links_a_wandb_run_that_a_record_carries` fails the
build if the README links a run no committed artifact backs.

---

## ⚡ Making It Go Faster, Measured

Everything above is about whether the detector is *right*. This section is about what it
costs to train, and every number in it was produced by a script in this repo on rented
A100s, not estimated.

### Gradient checkpointing was costing more than it was worth

`scripts/profile_train_step.py` runs the model step, forward through optimizer, at the
exact shapes `configs/training/baseline.yaml` uses: batch 32, sequence 512, bf16,
DeBERTa-v3-base. Two arms, one variable.

| Arm | Median step | p90 step | Peak memory |
|---|---|---|---|
| Gradient checkpointing on | 585.81 ms | 587.00 ms | 4.94 GiB |
| Gradient checkpointing off | 471.84 ms | 472.01 ms | 22.68 GiB |

Checkpointing costs **+24.15% step time** to save **17.74 GiB**. On an 80 GiB card that
is paying a quarter of the training budget for memory nothing was asking for, so the
full-scale configs now set it to `false` and say why in a comment. The minimal configs
keep it on deliberately: they are the reproduce-on-a-modest-GPU path, and there the
17.74 GiB is the whole point. A flag is not good or bad, it is a trade with a price, and
the price is now written down.

### The bottleneck is not the attention math

Ranked by self device time, checkpointing off:

| Operation | Share of device time |
|---|---|
| `aten::scatter_add_` | 20.87% |
| `aten::copy_` | 3.92% |
| `aten::bmm` | 2.34% |
| `aten::mm` | 2.13% |
| `aten::gather` | 1.5% |

The two matrix multiplies together are **4.47%**. One indexing op is **20.87%**, and the
`gather` it is the backward of is another 1.5%. That pair is DeBERTa-v3's disentangled
attention: the content-to-position and position-to-content relative indices read from a
shared bucket table, and the backward scatters gradients back into it, 72 times per
step. (`aten::scatter_add_` and the CUDA kernel beneath it both report 20.87%. That is
one bottleneck seen at two levels of the stack, not two.)

This is the reason `kernels/cuda/` is still empty. The plan was always to profile before
writing a kernel, and had I skipped that step I would have written a fused attention
kernel for something worth 4.47%. See the scope note at the end of the README for what
the kernel is now specified to do.

### FSDP shards, and the efficiency number is not free

`scripts/scaling_run.py` under `torchrun` on 1, 2 and 4 A100-SXM4-80GB. The global batch
is pinned at 64 at every world size, with `grad_accum` absorbing the difference, because
a scaling curve where each GPU count trains a different batch measures nothing.

| GPUs | Examples/s | Tokens/s | s/step | Peak GiB | Speedup | Efficiency |
|---|---|---|---|---|---|---|
| 1 | 65.03 | 33,298 | 0.9841 | 13.25 | 1.00 | 100.0% |
| 2 | 128.18 | 65,627 | 0.4993 | 11.85 | 1.97 | 98.5% |
| 4 | 254.54 | 130,325 | 0.2514 | 10.98 | 3.91 | 97.9% |

Peak memory *falls* as world size rises, from 13.25 to 10.98 GiB. That is the check that
FSDP is sharding parameters and not quietly replicating them, and it is worth more than
the throughput column.

> **The caveat, before anyone else finds it.** The 1-GPU baseline runs `grad_accum=4` to
> hold the global batch at 64, so it pays four micro-steps of overhead against the 4-GPU
> arm's one. That flatters the speedup. Holding the global batch fixed is the honest way
> to run this comparison, and this is what it costs. 97.9% is a real number with a known
> bias, not a clean one.


### The serving path is not batch-bound, and the code assumed it was

Everything above is training. Serving is a different machine: FORGE deploys on CPU in
float32, scoring 512-token windows. `scripts/benchmark_inference.py` sweeps batch size on
that path.

| Batch (windows) | Median | p95 | Windows/s | Samples |
|---|---|---|---|---|
| 1 | 452.7 ms | 472.6 ms | 2.21 | 7 |
| 2 | 848.2 ms | 890.9 ms | 2.36 | 7 |
| 4 | 1715.8 ms | 1991.4 ms | 2.33 | 7 |
| 8 | 3201.7 ms | 3363.4 ms | **2.50** | 7 |
| 16 | 6399.8 ms | 6916.9 ms | 2.50 | 4 |
| 32 | 15794.8 ms | 17571.6 ms | 2.03 | 3 |

Throughput is flat within 13% from batch 1 to 16. **Batching is not the lever on this
path.** It is a GPU optimisation, and this path has no GPU.

Worse, `scorer.py` hardcoded `batch_size=32` and `BatchPolicy` defaulted `max_batch=32`.
Batch 32 is the slowest point in the sweep, below batch 1, while making a single request
wait 15.8 seconds instead of 3.2. The code was paying five times the latency for negative
throughput. Neither number had ever been measured; both were the value that looks right
for a GPU. Both are now 8, and
[`tests/unit/test_serving_batch_size.py`](../tests/unit/test_serving_batch_size.py) reads the
committed artifact and fails if the constant and the measurement ever disagree again.

`forge/inference/batching.py` says its wait window is tuned "against the P95 latency
budget in the release gate". Until this sweep, no P95 had ever been measured, so two
modules were citing a budget that had no artifact behind it. There is one now.

#### Threads are the lever, and there are not many of them

The batch sweep above says what does *not* help. This says what does, at the batch size the
server actually uses, on the same 10-core machine:

| Threads | Windows/s | Speedup | Efficiency |
|---|---|---|---|
| 1 | 2.66 | 1.00 | 100.0% |
| 2 | 3.70 | 1.39 | 69.5% |
| 4 | 4.28 | 1.61 | 40.2% |
| 8 | 4.80 | 1.80 | 22.6% |
| 10 | 4.84 | 1.82 | 18.2% |

**Ten cores buy roughly 1.8x, and the curve is flat by four.** Torch's intra-op scaling on
this model is badly sublinear, which is the real ceiling on the serving path: not batch
size, not partitioning, just the fact that one 512-token encoder forward does not
parallelise well across cores.

That is independently corroborated by the Spark sweep below, which found two partitions of
five threads beating one partition of ten. Those two results only agree if ten threads buy
well under 10x, and they were measured for different reasons on different code.

> **The spread, stated rather than smoothed.** This sweep was run twice. The 1-thread
> baseline came out 2.29 then 2.66, a 16% difference, and the speedup at 10 threads came
> out 2.02 then 1.82. The baseline is the noisiest point by construction: longest
> per-sample time, so fewest samples inside the per-point budget, and it is the denominator
> for every other row. The rows at 4, 8 and 10 threads agree within 5%. So the honest claim
> is **1.8 to 2.0x**, not a third significant figure. A third run was not taken because it
> would move 1.9x to 1.9x. Both runs and a discarded contaminated one are recorded in
> `cpu_threads.json`.

> **Two caveats, before anyone else finds them.** `torch_threads` was 4 on the machine
> that produced this, torch's default rather than the machine's 10. That suspicion is now
> measured: the thread sweep above shows the default leaves about 1.8x on the table, so
> every latency figure in the batch table is understated by roughly that. The batch
> comparison itself holds, because every point ran under the same thread count. And batch 32 got 3 samples under
> the sweep's per-point time budget, so treat the size of its degradation as approximate
> and its direction as real. The artifact records the sample count per row for exactly
> this reason.

**Scope.** This times the model forward over pre-tokenised windows. Tokenisation,
windowing and HTTP are not included, so a real request costs this plus those.


### Ray Train, and what a single machine can honestly prove about it

Ray Train is not a third sharding strategy to put beside FSDP and DeepSpeed. Its job is to
**start and supervise** a distributed job: place the workers, build the process group, hand
each one its rank, tear everything down when one dies. It replaces `torchrun`, not FSDP. So
benchmarking it against the other two would be a category error, and a throughput table
with Ray in it would mislead.

`scripts/ray_train_launch.py` runs the same gradient-accumulation step under
`ray.train.torch.TorchTrainer`. On this machine, 2 workers, CPU, gloo:

```
world_rank=0, local_rank=0, node_rank=0   |   world_rank=1, local_rank=1, node_rank=0
backend: gloo        global_batch_size: 64        grad_accum: 2
```

**What that shows:** `ray_scaling_config`, which had been unit-tested for months and never
handed to Ray, produces output `TorchTrainer` accepts; the workers got distinct ranks in a
shared process group; and the global batch came back as **64**, the same value `torchrun`
produces. Changing who starts the workers did not change what they train on.

**What it does not show:** anything about throughput. CPU, gloo, one machine, a two-layer
toy model at sequence length 128. Ray earns its place on multiple machines, which this has
never run on.

> **The artifact this replaced was a fabrication, and the fix is the interesting part.**
> The first version wrote its "Ray placed the workers and built the process group" sentence
> as a hardcoded string, and filled the supporting fields with `metrics.get(...)`. Ray 2.58
> is Train V2, where `Result.metrics` does not exist, so every field came back `null` and
> the script wrote a confident claim backed by nothing and exited 0. The claim and the
> evidence were produced independently and nothing checked that the second supported the
> first. Now the evidence is validated before the sentence is written, the sentence is
> interpolated from those same verified fields, and
> [`tests/unit/test_ray_launch_evidence.py`](../tests/unit/test_ray_launch_evidence.py)
> refuses any committed Ray artifact whose fields are null. Every other finding in this
> repository was a mechanism that never ran; this one ran, produced nothing, and reported
> success.

### The flywheel, as a DAG

The recurring job in FORGE is the flywheel itself: scan the reserve pool, cluster the
failures, generate targeted mirrors, retrain, evaluate, gate. It is the only thing here
that genuinely earns an orchestrator, and
[`orchestration/dags/forge_flywheel.py`](../orchestration/dags/forge_flywheel.py) is it.

Four decisions in that file are not defaults, and each has a cost behind it.

**Every step is a `BashOperator` shelling out to the `forge` CLI.** Airflow's scheduler
re-imports every DAG file on a short loop, so `import torch` at module scope costs seconds
of CPU and a gigabyte of RSS *per parse cycle*, in the process whose job is scheduling. Two
tests pin it: one times the import, one parses the file with `ast` and inspects the import
statements. The second exists because the first version of that check searched the source
**text** and failed on the docstring explaining the rule. A check that greps source code is
a check a comment can break.

**`catchup=False`.** Airflow defaults this to `True`. A `@weekly` DAG deployed with a
`start_date` three months back immediately queues twelve backfill runs, each of which
retrains a model.

**`max_active_runs=1`.** Two concurrent rounds mine the same reserve pool. The mined-id
ledger exists to stop round two re-finding round one's failures, and that guarantee is void
if the rounds overlap.

**No retries on train or gate.** A training job that failed on resources fails the same way
at twice the cost. A gate that failed on the numbers will fail on the same numbers.

Artifact paths carry the run id, because a flat filename already caused one collision in
this repository today, and a scheduler turns occasional into routine.

> **What is claimed and what is not.** The DAG is implemented, parses under Airflow 3, and
> its 11 tests run in **their own CI job** rather than skipping in the main one, because
> `importorskip("airflow")` in a job that does not install Airflow is how a test file goes
> unrun for months. It has **never been run end to end by a scheduler**. The reserve pool
> it would mine now exists, 96 shards and 90,948 documents streamed from Common Crawl,
> fingerprint `88d6a7e70baf352b`, and the scan has been run over it by hand; what has not
> happened is Airflow triggering that scan on a schedule and acting on the result. This is
> a validated pipeline definition, not a pipeline with a run history.

### DeepSpeed against FSDP, on the same hardware, at the same global batch

`training/distributed.py` had generators for FSDP, DeepSpeed ZeRO-3 and Ray, all pure
functions, all tested, none of them ever run. FSDP was implemented and measured first. This
is the DeepSpeed half, on 2x L40S, with **FSDP re-run on that same pod** so the comparison
is between strategies rather than between two GPU models.

| GPUs | FSDP ex/s | DeepSpeed ex/s | FSDP peak | DeepSpeed peak |
|---|---|---|---|---|
| 1 | 58.63 | **64.25** | 13.25 GiB | **10.96 GiB** |
| 2 | 106.31 | **114.63** | 11.85 GiB | **10.00 GiB** |
| Speedup | **1.81** | 1.78 | | |
| Efficiency | **90.7%** | 89.2% | | |

**DeepSpeed is faster and lighter at both world sizes, and scales very slightly worse.** It
starts from a better single-GPU number, so its speedup column is smaller while its
throughput column is larger. Reading only the speedup would say FSDP scales better; reading
only examples per second would say DeepSpeed wins. Both are true, and neither alone is the
result.

DeepSpeed's own breakdown shows where the efficiency goes: `bwd_allreduce` is 14.96 ms at
one GPU and 21.26 ms at two. That growth is ZeRO-3's extra communication crossing PCIe.

> **The confound, which makes this a comparison of defaults rather than of sharding.** The
> FSDP arm uses torch's `AdamW`, unfused, under `torch.autocast`. The DeepSpeed arm uses
> the engine's JIT-compiled `fused_adam` with native bf16 and no autocast, because stacking
> autocast on top would cast twice. `optimizer_step` at **8.9 ms** for 184M parameters is
> that fused kernel. So this measures two realistic default configurations, which is what
> people actually run, and it does **not** isolate ZeRO-3 from FSDP. At one GPU, ZeRO-3
> shards across one device and therefore shards nothing, so the entire single-GPU gap is
> the optimizer and precision path rather than sharding.

**Interconnect, measured by accident.** The same FSDP code and the same global batch reach
**98.5%** efficiency at two A100-SXM4 and **90.7%** at two L40S, with identical peak memory
of 11.85 GiB. A100 SXM has NVLink; L40S does not, so every all-gather crosses PCIe. That
eight-point gap is the interconnect and nothing else.

Ray remains a config generator. `build_run` refuses it with a reason: Ray Train runs this
script as a worker rather than wrapping a model, so it does not belong behind that call.

Records: [`reports/experiments/scaling/`](reports/experiments/scaling), filed per device.

### The GPU saturates at 1024 tokens

The same inference sweep, on an L40S:

| Batch (windows) | Median | Windows/s |
|---|---|---|
| 1 | 15.16 ms | 65.95 |
| 2 | 15.31 ms | **130.67** |
| 4 | 32.19 ms | 124.26 |
| 8 | 75.34 ms | 106.18 |
| 16 | 148.0 ms | 108.11 |
| 32 | 299.1 ms | 106.99 |

**Batch 1 and batch 2 take the same wall time.** Batch 2 is nearly free and doubles
throughput; past that, time scales linearly with batch and throughput goes flat. The model
saturates a 48 GB datacenter GPU at **1024 tokens**. A model bound by its matrix multiplies
would keep gaining to batch 32 and beyond.

**That is the third independent measurement pointing at the same cause.** The profiler
ranked `aten::scatter_add_` at 20.87% of device time against 4.47% for every GEMM combined.
The CPU sweep found batching worth 13%. This finds batching worth nothing past batch 2. All
three are what you would expect from a model bound by **indexing and scatter**, which are
memory-bandwidth-bound and do not parallelise with batch, and none of them are what you
would expect from a GEMM-bound model. Three experiments, two devices, one conclusion, and
it is the conclusion that specifies the Phase 7 kernel.

GPU inference is **105.8 windows/s against 4.84 on CPU**, roughly 22x, and the serving path
still deploys on CPU because 2.5 windows/s is enough for the traffic it has.

> Two things not published from that run. The batch-1 p95 of 125.94 ms against a 15.16 ms
> median is CUDA context creation and cuDNN autotune on first call, not a latency tail. And
> the GPU thread sweep is flat within 0.6% from 1 to 128 host threads, as it must be, since
> host threads are not the resource doing the work; its efficiency column is arithmetically
> valid and meaningless, so the artifact records why it is omitted rather than printing
> 0.008 for someone to misread. The flat GPU curve is the control that makes the CPU thread
> result a property of the hardware rather than an artifact of the harness.

### Spark and Beam, over one scan, on a machine that cannot demonstrate either

The mining scan is the one job in FORGE shaped for a distributed runner: it reads a reserve
pool sized in millions, runs one independent forward per document, keeps the small fraction
the detector is confidently wrong about, and needs no shuffle, no join and no
cross-document state.

It is written once and distributed twice. `forge/hard_negative/scan_core.py` **is** the
scan: the round-robin shard partitioning, the fixed total document budget, the division of
cores among partitions, the per-worker model cache, and the refusals. `spark_scan.py`
hands it to PySpark `mapPartitions`; `beam_scan.py` hands it to a Beam pipeline. Neither
runner owns a decision that affects the result, which is the only reason the three tables
below can sit next to each other.

All three sweeps read the **same 6 shards, 60,000 rows, fingerprint `15737474bd03195a`**,
scan the same 40 documents, and use the same total intra-op thread budget: 10 threads in one
partition, 5 in each of two, 2 in each of four.

**Ideal speedup here is 1.00, not the partition count.** Adding a partition adds no
hardware, it re-slices the same ten cores. A speedup above 1.00 is not parallel efficiency,
it is evidence that torch intra-op scaling is sublinear and that several narrow workers beat
one wide one. There is deliberately no efficiency column; see the note below.

**Spark, local mode:**

| Partitions | Threads each | Docs/s (compute) | Speedup | Skew |
|---|---|---|---|---|
| 1 | 10 | 3.46 | 1.00 | 1.00 |
| 2 | 5 | 4.89 | 1.41 | 1.04 |
| 4 | 2 | 5.03 | 1.46 | 1.02 |

**Beam, `multi_processing`, the mode Spark local is comparable to:**

<!-- beam-mode: multi_processing -->

| Partitions | Model loads | Docs/s (compute) | Speedup | Skew |
|---|---|---|---|---|
| 1 | 1 | 2.63 | 1.00 | 1.00 |
| 2 | 2 | 3.98 | 1.51 | 1.03 |
| 4 | 4 | 5.18 | 1.97 | 1.02 |

**Beam, `multi_threading`, where every partition is a thread in one process:**

<!-- beam-threading-mode: multi_threading -->

| Partitions | Loads | Docs/s (compute) | Speedup | Skew |
|---|---|---|---|---|
| 1 | 1 | 2.70 | 1.00 | 1.00 |
| 2 | 1 | 6.18 | 2.29 | 1.04 |
| 4 | 1 | 5.34 | 1.98 | 1.02 |

**The ceiling is the machine, and both runners find it.** Spark tops out at 5.03 docs/s,
Beam at 5.18 under processes and 6.18 under threads. Skew stays between 1.02 and 1.04 in
every row, so the partitions are balanced and the host is simply full. That is the expected
result for one machine and it is not evidence for either framework.

**Where they disagree is fixed overhead, and reporting speedup alone would flatter Beam.**
At one partition Spark does 3.46 docs/s and Beam does 2.63, about 25% slower on identical
work. Beam's per-element path encodes every record through the Fn API and writes results to
a file sink; Spark's `mapPartitions` plus `collect` does not pay that. So Beam shows the
larger speedup (1.97 against 1.46) from the weaker baseline and arrives at roughly the same
absolute rate. The ratio is better and the throughput is not, which is the whole reason both
columns are published.

**Threads against processes is the clearest thing the port measured.** Under
`multi_threading` the model loads **once** at every partition count, because scan_core's
cache is a module-level dict shared by the worker threads, and startup stays flat at 5.5 to
6.7 seconds. Under `multi_processing` it loads once per worker, 1, 2 then 4 times, and
startup climbs from 10.1 to 26.5 seconds. For a 40-document job that fixed cost dominates,
which is why wall-clock throughput *falls* as partitions rise while compute throughput
climbs. Spark local behaves like the process mode, as it must: 8.8 to 24.7 seconds of
startup across the same sweep.

> **The column I removed, and the number that forced it.** These sweeps used to publish
> efficiency, meaning speedup divided by partitions. That is parallel efficiency only when
> each added partition adds hardware. Here the thread budget is constant, so the correct
> denominator is 1.0 and the old column was measuring the wrong thing in a plausible-looking
> way. The `multi_threading` run made it undeniable by returning **114.3%**, which is not a
> superlinear speedup, it is arithmetic proof of a wrong denominator. The artifacts now
> carry `ideal_speedup: 1.0` and a `speedup_semantics` string, and the claim checker fails
> if the word reappears near these tables.

> **The measurement I nearly published before that.** The first sweep left torch at its
> default 4 threads regardless of partition count and reported 1.72x at four partitions.
> That number was real and meaningless: the serial baseline used 4 of 10 cores while the
> parallel arms used more. Fixing it made the headline *worse*, because the baseline
> improved by 51% and the parallel arms by less. A speedup is a ratio, and a ratio is only
> as honest as its denominator. Twice now.

### The same sweep at fifty times the scale, where the speedup disappears

The three tables above scan 40 documents. That is a pilot, and a pilot at that size is
mostly startup. So the Spark sweep was run again over the reserve pool built from Common
Crawl in this repo, **96 shards, 90,948 rows, fingerprint `88d6a7e70baf352b`**, scanning
**2,000 documents** per partition count instead of 40.

| Partitions | Threads each | Docs/s (compute) | Speedup | Skew | Wall (s) |
|---|---|---|---|---|---|
| 1 | 10 | 2.094 | 1.00 | 1.00 | 972.1 |
| 2 | 5 | 2.092 | 0.999 | 1.008 | 974.2 |
| 4 | 2 | 2.279 | 1.088 | 1.055 | 898.2 |

**The 1.46x is gone.** At 40 documents four partitions looked 46% faster than one; at 2,000
they are 8.8% faster, and two partitions are a rounding error slower than one. Nothing about
the job changed between the two runs. What changed is how much of the measured interval was
fixed cost: at 40 documents the per-process model load and JVM startup are a large fraction
of the run, and slicing them across workers looks like a speedup. Scan fifty times as much
and the compute dominates, the constant thread budget binds, and the curve flattens to what
the hardware actually allows.

Throughput also falls, 3.46 docs/s to 2.09, because the two pools are not the same corpus.
The pilot reads `data/silver`; this reads real Common Crawl text, which is longer and more
varied, so there is more to score per document. The two absolute rates are not comparable
and are not being compared. What is being compared is the *shape* of the scaling curve on
one pool at two sizes, and the honest reading is that the earlier speedup column was
measuring startup amortisation and calling it parallelism.

**Beam, re-run over the same pool at the same 2,000 documents**, under `multi_processing`,
which is the mode Spark local is comparable to:

| Partitions | Model loads | Docs/s (compute) | Speedup | Skew | Startup (s) |
|---|---|---|---|---|---|
| 1 | 1 | 0.802 | 1.00 | 1.00 | 17.6 |
| 2 | 2 | 2.248 | 2.80 | 1.01 | 14.1 |
| 4 | 4 | 2.640 | 3.29 | 1.03 | 17.1 |

**Read the two tables side by side and the speedup column falls apart completely.** Beam
reports 3.29x where Spark reports 1.09x, on the same pool, the same documents, the same
host, the same scan code. Beam is not three times better at parallelising. Its
single-partition run does **0.802 docs/s against Spark's 2.094**, two and a half times
slower on identical work, and almost all of its apparent speedup is Beam climbing out of
its own hole. At four partitions the two runners land within 16% of each other, 2.640
against 2.279, which is the number that actually describes the machine.

This is the fourth time in this repository that a speedup column has turned out to be a
statement about its denominator, and it is the clearest case of the four, because here both
denominators are published and the reader can see the trick without being told. It is also
the reason the absolute throughput column is never dropped: a ratio alone would have made
Beam look like the better runner.

Where Beam's baseline goes is not a mystery. Its per-element path encodes every record
through the Fn API and writes results to a file sink, and at one partition that cost is
paid serially with nothing to overlap it. Spark's `mapPartitions` plus `collect` does not
pay it at all. Splitting the work is what lets Beam hide that overhead behind concurrent
workers, which is a real property of the runner and not a defect, but it is a property of
the *overhead*, not of the scan.

**What is still missing.** Beam's `multi_threading` mode has not been re-run at 2,000
documents, so the thread-against-process comparison remains a 40-document result and is
reported above as one. Nothing here has run on a cluster, on Dataflow or on Flink.

Records: [`reports/experiments/spark_reserve/`](reports/experiments/spark_reserve) and
[`reports/experiments/beam_reserve/`](reports/experiments/beam_reserve) for the
2,000-document runs, [`reports/experiments/spark/`](reports/experiments/spark) and
[`reports/experiments/beam/`](reports/experiments/beam) for the pilots.

**What the port was actually worth, which was not the numbers.** Expressing the same job on
a second substrate is what tested whether the job was separable from its runner. It was not:
everything that decided whether the scan finishes was sitting inside a module named after
Spark, and porting it is what forced `scan_core` to exist. The port also found three defects
the Spark path could not expose, each of which produced a plausible table rather than an
error. `--runner=DirectRunner` no longer means the DirectRunner: on Beam 2.76 it resolves to
a switching runner that prefers Prism, a subprocess that ignores the worker count the sweep
varies, so the runner is pinned to `FnApiRunner` by name. The per-worker model cache was not
thread safe, and under the threaded runner four partitions each loaded their own copy of a
184M parameter model with the cache present and doing nothing. And the first attempt to
*report* that was wrong in the opposite direction, because a load counter read before and
after acquiring the scorer credits the load to every thread that merely waited on the lock.

**So the claim this repository makes is narrow.** Both jobs exist, are tested, parallelise
without skew, refuse to mine from anything that is not the reserve pool, and refuse a pool
containing generated shards. Neither has run on a cluster, on Dataflow or on Flink, and
`data/reserve/` is not redistributed with this repository, though it exists and its
fingerprint is recorded. A single host re-slices
cores rather than adding them, so none of these tables is evidence that either framework
would help at 5M documents. The architecture argument is in
the README's scope section; these tables are evidence that the job runs
and scales the way its shape predicts, on two runners, over a pool whose identity is
recorded rather than assumed.

Records: [`reports/experiments/profile/`](reports/experiments/profile),
[`reports/experiments/scaling/`](reports/experiments/scaling),
[`reports/experiments/inference/`](reports/experiments/inference),
[`reports/experiments/spark/`](reports/experiments/spark) and
[`reports/experiments/beam/`](reports/experiments/beam).

---

## 🌐 Mining the Internet, Actually Run

The reserve pool this project's flywheel needs did not exist until this run. It does now: **5,638,371 Common Crawl documents streamed and cleaned in 2.05 hours**, on a laptop, at 762 documents per second across 4 Spark workers. Artifact: [`reserve report`](../reports/experiments/crawl/spark/ab6890f623a36785-p4-s32.json).

| | |
|---|---|
| Crawl | CC-MAIN-2026-34, plan fingerprint `ab6890f623a36785` |
| Segments | 270 read of 284 planned |
| Documents streamed | 5,638,371 |
| Kept by the cleaning policy | 96,039 (1.70%) |
| Exact duplicates removed | 5,091 |
| Already in the training corpus | 0 |
| **Written to the reserve pool** | **90,948** (72,803 train, 9,041 val, 9,104 test) |
| Corpus fingerprint | `cfb144546adc2165` |

### What the rejection breakdown says

| Reason | Documents | Share of everything streamed |
|---|---|---|
| `too_long_tokens` | 3,325,856 | 59.0% |
| `language` | 1,269,280 | 22.5% |
| `too_short_chars` | 637,917 | 11.3% |
| `too_short_tokens` | 298,779 | 5.3% |
| `repetitive_lines` | 9,050 | 0.2% |

**The 400-token cap is the dominant filter, not language.** `too_long_tokens` removes 59% of the crawl against 23% for non-English. That cap comes from v0.1-min, where 400 tokens is exactly one 512-token window, and it is the recorded limitation that makes FORGE v1 a single-window detector. It was worth measuring rather than assuming: an earlier probe run under the DEFAULT cleaning policy reported a 9.6% keep rate, against 1.70% under the policy the models were actually trained on. Building the reserve pool under the wrong one would have filled it with documents longer than anything the detector has seen, and every false positive it produced would have been measuring that instead of the detector.

**Zero of 5,638,371 documents matched a training hash.** That is measured, not assumed: every content hash was checked against all 60,000 in `data/silver`. It is weaker evidence than it looks, because the check is exact-hash after normalisation and redaction, so a recrawl of the same page with one changed byte does not match. It rules out the crude failure and not the subtle one.

### Fourteen segments failed, and the run finished anyway

All 14 failures were HTTP 503 from data.commoncrawl.org, and all of them fall in a contiguous block of segment numbers, which is a server-side window rather than 14 independent faults. Each one was retried four times with backoff to a minute before being recorded.

That design is deliberate. An earlier version raised on the first failure and lost 37 minutes of completed work; the version before that lost 22. Losing 270 good segments to 14 bad ones is the worse outcome, so the run continues and the failures are listed per segment with their URLs, the CLI exits non-zero, and the report records `segments_read` as 270 rather than 284. A corpus assembled from fewer segments than were planned cannot be reproduced from the plan alone, and the artifact has to say so.

### And then the flywheel was pointed at it, which is what the pool was for

Building the pool was half the exercise. The other half is the mining stage actually running
against it: score real web text with the deployed model at the deployed threshold, and keep
whatever it gets wrong. That round has now run. Artifact:
[`reserve_round_001.json`](../reports/experiments/reserve_round_001.json).

| | |
|---|---|
| Model | `forge_min_mirror@8e06099f` |
| Documents scanned | 10,000 |
| Operating threshold | 0.998252 |
| Human documents scored above it | **2** |
| Measured false-positive rate | **0.020%** |
| FPR budget the threshold was fitted to | 0.1% |
| Selected into the round | 2 (1 train, 1 holdout) |

**The interesting number is the one that is small.** The threshold was fitted on a validation
split to hold false positives under 0.1%. On 10,000 documents of real Common Crawl text,
which is not that validation split and shares no generator, register or crawl with it, the
rate came out at 0.020%, five times inside budget. That is the first evidence in this project
that the operating point survives contact with the open web, and it is worth stating clearly
because almost every other generalisation result here goes the other way: the detector misses
most AI text it has not seen, and it does that while almost never accusing a human. Those are
different failures and only one of them is the one a user gets hurt by.

**The mining half of the round is not a result, and the artifact says so itself.** Two
failures is not a corpus. The round still ran its clustering stage over them and produced two
clusters of one document each, so it recorded its own warnings rather than a silhouette
score:

> `embedder is hashing_v1, which captures surface form but not semantics. Fine for tests, not for a real mining run.`
>
> `clusters share a metadata label ['informational / web / via cc']: one true mode was likely split across several clusters, which over-weights it in proportional selection.`

Both are correct and both were written by the pipeline, not by me afterwards. At n=2 the
atlas, the proportional per-cluster quota and the train/holdout split are arithmetic over
noise. What the round demonstrates is that the stage runs end to end on a real corpus, keeps
train and holdout separate, and refuses to flatter itself. What it does not demonstrate is
that mining improves the model, because two documents cannot.

**The honest next step, which has not run.** At 0.020%, finding enough false positives to
retrain on means scanning on the order of a million documents rather than ten thousand, which
the pool has enough shards for at roughly 2.5 documents per second per partition, so about a
day of laptop time. That is the experiment. The flywheel exists and turns; it has not yet
turned far enough to move anything.

---
## ⚙️ ONNX Runtime: a serving path that did not go faster

vLLM does not fit this model. FORGE-Base is a bidirectional encoder over a fixed 512-token window with no KV cache, so continuous batching and paged attention have nothing to work with. ONNX Runtime does fit it, in principle: graph fusion and int8 kernels target exactly the per-window encoder compute this serving path pays for. So it was worth measuring. It did not pay off.

| Intra-op threads | PyTorch windows/s | ONNX Runtime windows/s | Speedup |
|---|---|---|---|
| 1 | 2.66 | 2.70 | 1.02x |
| 2 | 3.70 | 3.84 | 1.04x |
| 4 | 4.28 | 4.48 | 1.05x |
| 8 | 4.80 | 4.80 | 1.00x |
| 10 | 4.84 | 4.56 | 0.94x |

Batch 8, float32, same windows, same scope as [`cpu_latency_b8_t*.json`](../reports/experiments/inference): model forward only, no tokenisation or HTTP. Artifact: [`onnx_summary_mirror.json`](../reports/experiments/inference/onnx_summary_mirror.json).

**Two to five percent at low thread counts, nothing at 8, and slower at 10.** At 512 tokens this encoder is bound by memory bandwidth rather than by operator dispatch, so the graph fusions ONNX Runtime brings have little left to remove, and at 10 threads its own scheduling costs more than it saves. Both runtimes plateau at the same place, which is the machine.

**The numbers ARE the same model.** Maximum absolute difference across the parity sample is 2.28e-10 and 0 verdicts changed. The flip count alone would be weak evidence, because the sample happened to contain 0 windows near the 0.998252 threshold and a sample that could not flip proves nothing. The delta is what carries it: at 1e-10, a flip would need a window sitting within 1e-10 of the threshold.

**int8 is where the speedup would have been, and it does not build.** Dynamic quantisation fails in onnxruntime's shape inference on the graph torch's dynamo exporter produces:

```
InferenceError: [ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (768) vs (2)
```

That is a toolchain incompatibility, not a property of the model, and it is the honest reason the CPU serving path stays float32. Quantisation was also the only variant that could have moved the verdicts, which is why the script measures flips rather than assuming them away.

**What this row claims, therefore:** an ONNX export exists, it is numerically the deployed model, and it is measured against the PyTorch path on the same windows. It is not a speedup. Reporting it as one would have required either not measuring the baseline or not publishing the comparison.

---
## 🛡️ Adversarial: what breaks the detector, and what fixes it for free

Four preprocessing conditions, each scored against **its own** clean baseline, because folding and casefolding move clean documents too and a defence that lowers the attacked miss rate by moving everything is not a defence. 250 AI documents per arm, 1,000 human documents for the cost side. Artifacts: [`adversarial_forge_min_baseline.json`](../reports/experiments/adversarial_forge_min_baseline.json), [`adversarial_forge_min_mirror.json`](../reports/experiments/adversarial_forge_min_mirror.json).

**forge_min_baseline@8e06099f**, threshold 0.996586

| Attack | Severity | raw | normalised | folded | casefolded |
|---|---|---|---|---|---|
| `homoglyph_substitute` | 0.2 | 0.316 | 0.316 | 0.000 | 0.000 |
| `case_perturb` | 0.02 | 0.016 | 0.016 | 0.016 | 0.000 |
| `case_perturb` | 0.1 | 0.980 | 0.980 | 0.980 | 0.000 |

Human score distribution over 1,000 documents (the FPR column is 0.0000 in every condition, which at 1/1000 resolution says almost nothing; the distribution is what carries the information):

| Condition | median | p95 | p99 | max |
|---|---|---|---|---|
| raw | 0.000025 | 0.000075 | 0.000169 | 0.0267 |
| normalised | 0.000025 | 0.000075 | 0.000169 | 0.0267 |
| folded | 0.000025 | 0.000075 | 0.000169 | 0.0267 |
| casefolded | 0.000021 | 0.000042 | 0.000066 | 0.0013 |

**forge_min_mirror@8e06099f**, threshold 0.998252

| Attack | Severity | raw | normalised | folded | casefolded |
|---|---|---|---|---|---|
| `homoglyph_substitute` | 0.2 | 0.252 | 0.252 | 0.000 | 0.000 |
| `case_perturb` | 0.02 | 0.012 | 0.012 | 0.012 | 0.000 |
| `case_perturb` | 0.1 | 0.876 | 0.876 | 0.876 | 0.000 |

Human score distribution over 1,000 documents (the FPR column is 0.0000 in every condition, which at 1/1000 resolution says almost nothing; the distribution is what carries the information):

| Condition | median | p95 | p99 | max |
|---|---|---|---|---|
| raw | 0.000043 | 0.000232 | 0.002872 | 0.9769 |
| normalised | 0.000043 | 0.000232 | 0.002872 | 0.9769 |
| folded | 0.000043 | 0.000232 | 0.002872 | 0.9769 |
| casefolded | 0.000037 | 0.000183 | 0.001275 | 0.7915 |

### Three readings

**Folding confusables is free, and it closes the homoglyph attack completely.** The `folded` column is identical to `normalised` at every statistic on both arms, so it costs nothing, and it takes `homoglyph_substitute` at severity 0.2 to a miss rate of 0.000. NFKC alone does not: Cyrillic а and Latin a are different characters that NFKC is correct to leave alone, and TR39 confusable folding is the step that maps them together.

**Casefolding is not score inflation, and I expected it to be.** My hypothesis was that its perfect zeros came from pushing every score up. They come from pushing the human distribution DOWN on both arms (baseline max 0.0267 to 0.0013, mirror 0.9769 to 0.7915). The hypothesis was wrong twice: first at 500 human documents where the instrument had no power, then again at 1,000 where it did. It stays unrecommended until the out-of-distribution sweep is re-run with the threshold refitted, because a transform that moves every score needs a refitted threshold before it means anything.

**The mirror arm has a human document at 0.9769 against a threshold of 0.998252.** The baseline arm's worst human document scores 0.0267, nowhere near its own threshold. Same 1,000 documents, same conditions. The mirror arm is the one that wins on out-of-distribution AUROC, and this is what that win costs: its margin against a false positive is about a hundredth, not a whole number. That is a property of the arm, not of any attack, and it belongs beside the AUROC whenever the AUROC is quoted.

---
## 🖥️ The Interface

Two shells over the same detectors: a FastAPI reference page and a Streamlit page that is
what deploys. Both render the same cards from the same payload against the same stylesheet,
so they cannot drift into disagreeing with each other. The captures below are from the
reference page.

<!-- The text-tab capture is deliberately absent. The only one ever taken was at commit
4ec8204c against threshold 0.996956, and those checkpoints were retracted when the chat
template defect was found. A screenshot of withdrawn numbers is worse than no screenshot.
A fresh capture goes here once the live demo serves the current checkpoints. -->

<br/>

<div align="center">
<img src="../images/image_verdict.png" width="92%" alt="Image tab: verdict, evidence and file signals"/>
<br/>
<em>Declaration-first verdict logic. The detector's own probability leads the evidence panel,
supporting signals report a word rather than a percentage, and nothing here is combined into
a single invented score.</em>
</div>

<br/>

<div align="center">
<img src="../images/image_robustness.png" width="92%" alt="Robustness across eleven transforms"/>
<br/>
<em>Eleven edits an image meets in the wild, each re-scored and compared against the
original. "flipped" means that transform changes the answer. This asks whether the VERDICT
survives redistribution, which is a different question from whether a forensic signal does.</em>
</div>

<br/>

<div align="center">
<img src="../images/image_attribution.png" width="92%" alt="Occlusion attribution"/>
<br/>
<em>Occlusion attribution in the model's own preprocessed tensor space: each region is
hidden and the image re-scored, so a warm cell is one the decision actually rested on.
Measured, not gradient-approximated, so it holds for any detector the project loads.</em>
</div>

### Three design rules the interface never breaks

| Rule | Why |
|---|---|
| **"NO AI DETECTED", never "HUMAN"** | A detector cannot establish that a person wrote something. It reports whether the input resembles the distribution it was trained on. One is a claim about the world, the other about the model. |
| **A declaration outranks a probability** | An IPTC `trainedAlgorithmicMedia` tag or a Stable Diffusion parameter block is a label the generator wrote about its own output. A mediocre detector does not get to overrule it. |
| **Absence is reported as absence** | No EXIF means the file has no EXIF. It does not mean AI. Metadata-derived scores fire hardest on screenshots and re-saved photographs, which is precisely the false accusation this project exists to prevent. |

---

## 🏗️ Pipeline

```
FineWeb / FineWeb-Edu
        │
        ▼
┌────────────────────────────────────────────────┐
│  Ingestion & cleaning                          │
│  • language id · PII scrub · normalisation     │
│  • MinHash + exact dedup                       │
│  • source-group splits (no leakage across arms)│
└───────────────────┬────────────────────────────┘
                    │  FORGE-HUMAN corpus
        ┌───────────┴───────────┐
        ▼                       ▼
┌────────────────┐      ┌────────────────────────┐
│  ARM A         │      │  ARM B                 │
│  random prompts│      │  mirror engine         │
│                │      │  • attribute extraction│
│                │      │  • generator pinning   │
│                │      │  • length matching     │
└───────┬────────┘      └───────────┬────────────┘
        │                           │
        └───────────┬───────────────┘
                    ▼
┌────────────────────────────────────────────────┐
│  Training · DeBERTa-v3-base, bf16, RTX 4090    │
│  • windowed scoring, mean-pooled               │
│  • checkpoint chosen by FNR *inside* the       │
│    FPR budget, not by the budget itself        │
│  • temperature calibration on validation       │
└───────────────────┬────────────────────────────┘
                    ▼
┌────────────────────────────────────────────────┐
│  Evaluation                                    │
│  • in-distribution + HC3 / MAGE / RAID         │
│  • paired bootstrap · McNemar at matched budget│
│  • calibration under shift (ECE)               │
│  • every run record committed to reports/      │
└───────────────────┬────────────────────────────┘
                    ▼
        Serving · CPU, float32, two shells
        FastAPI reference page · Streamlit deploy
```

---

## 🔧 Components

### `src/forge/generation/mirror.py` — the treatment
Extracts attributes from a human document (topic, length, register, structure), pins a
generator, and produces a synthetic counterpart matched on all of them. This is the single
variable the experiment manipulates.

### `src/forge/training/train.py` — checkpoint selection that actually selects
The original selector scored checkpoints by FPR-at-budget, which is **pinned by construction**
to the budget: every epoch produced the same number, `<=` handed the win to the last one, and
`best.pt` was byte-identical to `last.pt`. Selection now prefers a checkpoint inside the
budget and breaks ties on FNR, the thing the project is trying to minimise.

### `src/forge/evaluation/ood.py` + `scripts/ood_mcnemar.py` — the statistics
The first significance test was a two-proportion z-test on FNRs measured at **two different
thresholds**, treating paired data as independent samples. It reported p = 0.0 on MAGE. The
paired McNemar at a matched budget reports **p = 0.243**. The replacement changed a headline
claim, which is the point of doing it properly.

### `src/forge/inference/scorer.py` — the CPU serving path
Loads an arm, windows the document, mean-pools. Contains one line that matters more than it
looks: `model.float()`. Training ran in bf16, and CPU matmul refuses to mix Half and Float,
so without it *no text scored at all* on the only hardware this path targets.

### `src/forge/image/detector.py` — polarity resolved by measurement
Which class a third-party model calls "AI" is read from labelled images, never from its
`id2label` documentation. A detector whose polarity is unverified reports that it is
unverified instead of producing a possibly inverted verdict.

### `src/forge/image/attribution.py` — occlusion, not saliency
Hides each region of a 5×5 grid, re-scores in one batched forward pass, and maps the drop in
AI probability. Measured rather than gradient-approximated, so it holds for any detector the
project loads.

### `src/forge/ui/` — one template, two shells
The FastAPI page and the Streamlit page render the same cards from the same payload against
the same stylesheet. Two renderers over one payload is how pages drift into disagreeing with
themselves, so there is exactly one of each.

---

## 🧪 What the Test Suite Is For

**1157 tests**, and the interesting ones are not unit tests. They are regression tests, each
named after a specific wrong answer this project shipped and then caught:

| Test | The failure it locks out |
|---|---|
| `test_a_half_precision_checkpoint_still_scores_on_cpu` | bf16 weights on CPU: no text scored at all |
| `test_the_best_checkpoint_is_not_simply_the_last_one` | selection metric pinned by construction |
| `test_a_phone_photograph_in_an_mpo_container_is_analysed_not_rejected` | an allowlist refusing real Canon photographs |
| `test_no_shipped_string_says_the_visual_detector_does_not_exist` | the payload denying a detector that had just scored |
| `test_the_serve_extra_can_actually_serve` | an install that booted with no model stack |
| `test_the_gauge_marks_the_threshold_the_verdict_actually_uses` | a gauge implying a 50% boundary against a 0.992 threshold |

The pattern behind every one of them is the same: **something reported a state it was not
measuring**. That is also the failure mode detection products die of, which is why the tests
are written as narratives rather than assertions.

---

## 🚀 Run It

### Live
**https://panagramforge-cqzwwskdjhbfv6hxwppvkz.streamlit.app**

### Locally
```bash
git clone git@github.com:AKilalours/Forge_Panagram.git
cd Forge_Panagram
python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,image,dev]"
```

```bash
# The reference interface
python -m uvicorn api.forge_app:app --port 8000

# The deployed interface
streamlit run streamlit_app.py

# Everything, including the regression suite
python -m pytest -q
```

Text checkpoints are fetched from
[`Akilalourdes/forge-detect-weights`](https://huggingface.co/Akilalourdes/forge-detect-weights)
on first use. Serving is CPU-only; no GPU is required to reproduce any inference result.

### Reproducing the evaluation without a GPU
```bash
python scripts/eval_ood.py --arm mirror --benchmark raid
python scripts/ood_mcnemar.py
python scripts/ood_table.py
```
Every figure in this README is read from a committed record under `reports/experiments/`.
Nothing here is typed in by hand.

---

## 🔬 Tech Stack

| Layer | Technology |
|---|---|
| Backbone | `microsoft/deberta-v3-base` |
| Training | PyTorch, bf16, NVIDIA RTX 4090 on RunPod, ~21 min per arm |
| Scaling and profiling | FSDP (`FULL_SHARD`) under `torchrun`, `torch.profiler`, 4x A100-SXM4-80GB on RunPod |
| Corpus | FineWeb / FineWeb-Edu, MinHash dedup, source-group splits |
| Benchmarks | HC3 · MAGE · RAID, evaluation-only |
| Statistics | paired bootstrap (10k resamples), McNemar at a matched budget |
| Serving | FastAPI · Streamlit · CPU float32 |
| Image forensics | Pillow, C2PA / IPTC parsing, ELA, resampling, quantisation tables |
| Weights | Hugging Face Hub |

---

## 📁 Structure

```
Forge_Panagram/
├── src/forge/
│   ├── generation/         # mirror engine: the experiment's one variable
│   ├── training/           # training loop, checkpoint selection, calibration
│   ├── evaluation/         # OOD harness, metrics, calibration, release gate
│   ├── inference/          # CPU scorer, decision policy, shared text API
│   ├── image/              # detector, forensics, occlusion attribution
│   ├── ui/                 # ONE stylesheet and ONE set of result cards
│   ├── dedup/ cleaning/    # MinHash, exact dedup, PII, language id
│   └── failure_atlas/      # clustering of what the detector gets wrong
├── api/forge_app.py        # FastAPI reference interface
├── streamlit_app.py        # deployed interface
├── scripts/                # eval_ood · ood_mcnemar · image_detector_probe · …
├── reports/experiments/    # every committed run record and score array
├── docs/                   # evaluation · writeup · data spec
├── demo/                   # held-in AI samples for testing the text tab
└── tests/unit/             # 1157 tests, most named after a real bug
```

---

## ⚠️ Honest Scope

| Claim | Status |
|---|---|
| Two arms trained and compared under one changed variable | ✅ done, records committed |
| Mirroring helps at low FPR on HC3 and RAID | ✅ measured, McNemar at matched budget |
| Mirroring helps on MAGE | ❌ **no effect**, p = 0.243 |
| Either arm is deployable out of distribution | ❌ **no**, 63–96% miss rate |
| Image side is a trained FORGE model | ❌ **no**, a measured third-party baseline |
| Image operating point is validated | ❌ fitted in sample on 20 generated and 9 human images |
| Arms C (hard negatives) and D (adversarial) | ⏳ wired, not yet trained |

---

## 💡 One-Liner

> *"Trained two DeBERTa-v3 detectors differing only in how their synthetic half was
> generated, and tested the difference properly: paired bootstrap and McNemar at a matched
> 0.1% false-positive budget. Mirrored synthetic data cut the HC3 miss rate from 94% to 63%
> and caught 122 RAID documents the control missed, while doing nothing measurable on MAGE.
> Both arms remain undeployable out of distribution, and the shipped interface says so."*

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0d2b,50:1a1a4e,100:0f0c29&height=120&section=footer" width="100%"/>

**© 2026 Akila Lourdes Miriyala Francis**

*DeBERTa-v3 · PyTorch · FastAPI · Streamlit · HC3 · MAGE · RAID · FineWeb*

</div>
