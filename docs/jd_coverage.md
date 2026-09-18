# JD coverage matrix

Where each required capability lives in FORGE, which phase implements it for real, and
what would make the claim defensible in an interview.

The rule this table follows: a row is only marked **implemented** when there is code
that runs and a measurement committed alongside it. Everything else says *planned*
with the phase that owns it. An interviewer who opens the repo should find the state
exactly as described here.

**Scope.** This table describes FORGE only. Two of the requirements it marks absent are
met elsewhere in my work rather than in this project: Airflow (an ETL pipeline into
Snowflake with CDC-style incremental upserts at Archimedis Digital, and a training and
release DAG that blocks promotion on a failed check in StreamLens) and GCP (a production
BigQuery pipeline unifying 14+ event sources with de-identification applied at ingest, at
Jaan Health). Stub files standing in for those here would have been worse than nothing,
so they were removed rather than left raising.

| # | Requirement | Where in FORGE | Phase | State |
|---|---|---|---|---|
| 1 | Python and modern ML frameworks | whole `src/forge` tree, PyTorch + HF Transformers | 3 | scaffolded |
| 2 | Transformers and LLM fundamentals | `modeling/encoder.py` (encoder + dual head), `modeling/windowing.py` (overlapping windows), `generation/` (decoding grid, sampling) | 3 | windowing implemented and tested; model planned |
| 3 | Research and engineering boundaries | `docs/data_spec_v1.md` (frozen contract), `evaluation/release_gate.py`, `configs/eval/regimes.yaml` | 0 | **implemented** |
| 4 | NVIDIA GPU programming and CUDA | `training/profiling.py` + `scripts/profile_train_step.py`, run on an A100-80GB. Ranked device time committed at `reports/experiments/profile/`. The target is named by measurement, not by guess: `aten::scatter_add_` is **20.87%** of device time, against **4.47%** for `bmm` and `mm` combined. `kernels/cuda/` is still empty. | 7 | profiling **implemented and run**; kernel **not written** |
| 5 | Distributed training (DeepSpeed, FSDP, Ray) | **All three run, at different levels.** FSDP on 1/2/4 A100-SXM4 at 98.5% and 97.9%; FSDP vs DeepSpeed ZeRO-3 head to head on 2x L40S at a fixed global batch of 64, where DeepSpeed is 8% faster and 16% lighter while scaling 1.5 points worse. Ray Train launches the same job via `TorchTrainer` (`scripts/ray_train_launch.py`), verified on CPU with gloo at 2 workers: **a wiring check, not a benchmark**. `DistributedRun.micro_step` owns the step semantics so the strategies cannot be compared unfairly by accident. | 7 | FSDP + DeepSpeed **measured on GPU**; Ray **launches correctly on CPU**, never on GPU or multi-node |
| 6 | Inference frameworks (vLLM) | `generation/generators/base.py`: two-pass scheduler holding one engine at a time, GPU preflight, explicit allocator teardown, scheduler invariants under test. Single GPU. | 2 | **implemented** |
| 7 | Large-scale data processing (Spark, Beam) | **Both, over one scan.** `hard_negative/scan_core.py` is the reserve-pool mining scan: round-robin shard partitioning, model loaded once per worker, a fixed total document budget split across partitions, and a strict refusal to mine from non-reserve roots. `spark_scan.py` distributes it with PySpark `mapPartitions`; `beam_scan.py` distributes it with a Beam pipeline. Neither owns a decision that affects the result, which is what makes the two measurements comparable. The Spark sweep is recorded at `reports/experiments/spark/`; the Beam sweep is not recorded yet, so this row claims the port and not a number. Cleaning stays on Polars, deliberately. | 7 | Spark **implemented and run locally**; Beam **implemented and tested, sweep not yet recorded**; **neither on a cluster or a distributed runner** |
| 8 | Orchestration (Airflow) | `orchestration/dags/forge_flywheel.py`: the mining flywheel as a DAG. BashOperator throughout so the scheduler never imports torch at parse time; `catchup=False`, `max_active_runs=1`, no retries on train or gate; every artifact path scoped to the run id. 11 tests in their own CI job so they cannot silently skip. | 8 | DAG **implemented and validated in CI**; **never run against a real corpus** |
| 9 | MLOps and experiment tracking | `registry/model_registry.py`, W&B config in every training YAML, `MANIFEST.json` dataset versioning | 3 | registry contract implemented |
| 10 | DevOps tools | `.github/workflows/ci.yml` (lint, tests, spec check, README claim check), `Makefile`, `infra/docker/`, `pyproject.toml`, ruff/mypy/pytest | 0 | **implemented**, and the workflow now actually triggers: it was pinned to a `main` branch this repo does not have |
| 11 | Cloud infrastructure (AWS/GCP) | `infra/docker/`, `infra/terraform/` (README only), S3/MinIO storage layout in spec section 8. Compute ran on RunPod; the app deploys on Streamlit Community Cloud. | 7 and 8 | planned |

---

## The honest version of each claim

**CUDA (#4).** The rule was: profile first, then write the kernel the profile asks
for. The profile has now been run, on an A100-80GB at the shapes
`configs/training/baseline.yaml` actually uses, and it gave an answer I did not
expect. `aten::scatter_add_` is **20.87%** of device time. Its CUDA kernel,
`_scatter_gather_elementwise_kernel`, is the same 20.87% seen one level down, so that is
one bottleneck and not two. The matrix multiplies, `bmm` at 2.34% and `mm` at 2.13%, are
**4.47%** together. The attention math is not the bottleneck; the *indexing around it*
is. That op is the backward of the `gather` in
DeBERTa-v3's disentangled attention, where the content-to-position and
position-to-content relative indices gather from a shared bucket table, and the
backward scatters gradients back into it with `ReduceAdd` over 72 calls per step.

So the kernel is now specified rather than speculated about: fuse the c2p/p2c index
construction and its scatter-add so the bucket table is written once per layer instead
of gathered and scattered twice. That is the Phase 7 deliverable, and it is not written
yet. What is committed is the measurement that names it, at
`reports/experiments/profile/train_step_comparison.json`. Writing a CUDA kernel for a
stage that is 3 percent of step time is resume decoration, not engineering, and it
would have been exactly what I did without this profile.

**vLLM (#6).** vLLM belongs on the generation side, where FORGE decodes hundreds of
thousands of documents from open-weight models and continuous batching genuinely
helps. It does not belong on the detector's serving path: FORGE-Base is a
bidirectional encoder with a fixed 512-token window and no KV cache, so vLLM's core
optimizations do not apply. Being able to explain *why not* is worth more than a
misapplied dependency.

**Spark / Beam (#7).** Phase 1 runs on Polars and PyArrow because 400k documents fit
on one machine and Spark would add operational cost for no throughput. That is still the
right call and the cleaning path has not changed.

The mining scan is the job with a different shape, and it is now written twice:
`hard_negative/scan_core.py` is the scan, `spark_scan.py` and `beam_scan.py` are two ways
of distributing it. They live with the Phase 4 mining code rather than in a folder that
exists to name a framework. The scan reads sharded parquet, loads the detector once per
worker, scores each document independently, and returns only the confident false positives
so the pool does not cross the network to be discarded.

**What the second runner was actually for.** Not a benchmark. The scan had been written for
Spark, and everything that decided whether it finished had ended up inside a module named
after Spark: the shard partitioning, the document cap that holds the work constant, the
division of cores among partitions, the per-worker model cache. Porting it was the test of
whether any of that was really Spark's, and the answer was no, which is why `scan_core`
exists. The port also found two defects the Spark path could not expose, both of which
produced a plausible table rather than an error, and both of which are written up in
`beam_scan.py`: `--runner=DirectRunner` no longer means the DirectRunner, and the
per-worker model cache was not thread safe. Both were found on a stub scorer, not on the
real arms, so the Beam sweep on this machine is still unrecorded and this section makes no
throughput claim for it.

**What the local run proved, and what it could not.** On a 10-core machine, scanning a
fixed 40 documents at 1, 2 and 4 partitions, throughput tops out near 5.5 docs/s however
the cores are divided, with skew at 1.03. The partitions are balanced; the host is full.
That is the expected result and it is not evidence for Spark: a single machine
redistributes cores rather than adding them. The job has never run on a cluster, and
`data/reserve/` is empty because the corpus is not redistributed, so the 5-million-document
pool it is built for does not exist here.

**The number I nearly reported.** The first sweep left torch at its default 4 threads in
every arm and showed 1.72x at four partitions. The serial baseline was using 4 of 10 cores
while the parallel arms used more, so the speedup was an artifact of a handicapped
denominator. Correcting it lowered the headline to 1.31x. Being able to explain why the
worse number is the true one is the part of this row worth interviewing about.

**Airflow (#8).** The recurring pipeline is the flywheel itself: scan reserve pool,
cluster failures, generate targeted mirrors, retrain, evaluate, gate. That is a real DAG
with real dependencies and retry semantics, so it is worth orchestrating. A DAG that runs
one script on a schedule is not.

This repo previously held a file listing those task names and a schedule string, with no
DAG object and no callables. That is a sketch, and a sketch that sits in a directory
called `orchestration/airflow/dags/` reads as a capability rather than a plan, so it has
been removed. The Airflow experience behind my CV is in the two production pipelines
named in the Scope note above.

**AWS/GCP (#11).** Local development uses MinIO with the identical S3 layout from spec
section 8, so the cloud move is a config change rather than a rewrite. Terraform
defines the training bucket, the GPU instance profile and the registry. The measured
claim to aim for is a cost-per-training-run number, which is far more convincing than
a list of service names.

**Experiment tracking (#9).** The claim worth making is not "used W&B". It is that
every run is reproducible from `dataset_version` plus `code_commit` plus config, and
that `registry/model_registry.py` refuses an entry missing any of them. That is
already enforced by a test.

## Why the detector is not served with vLLM

vLLM's value is continuous batching and a paged KV cache for autoregressive decoding.
FORGE-Base is a bidirectional encoder with no KV cache and a fixed 512-token window, so
none of that machinery applies to it. vLLM is used on the GENERATION side instead, in
`src/forge/generation/generators/base.py`, where the workload actually is autoregressive
decoding at volume.

The detector is served by FastAPI (`api/forge_app.py`) and Streamlit
(`streamlit_app.py`). If the latency budget ever forces it, the next steps are dynamic
batching, then ONNX Runtime if it measurably wins, then TensorRT only if the numbers
justify the build complexity. None of those are in place, because nothing has measured
that they are needed.
