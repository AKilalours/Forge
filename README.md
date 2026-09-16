<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:1a1a4e,100:0d0d2b&height=200&section=header&text=FORGE%20%F0%9F%94%8D&fontSize=58&fontColor=ffffff&fontAlignY=38&desc=Failure-Driven%20Synthetic%20Data%20for%20Robust%20AI-Content%20Detection&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>

### *Built by* **Akila Lourdes Miriyala Francis**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-CPU%20%2B%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/DeBERTa--v3-base-FFB000?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/RunPod-RTX%204090-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Tests-951%20passing-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/In--distribution%20AUROC-0.99997-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FPR%20budget-0.1%25-0056D2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Headline-Partial%20null%20result-FF8F00?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://panagramforge-cqzwwskdjhbfv6hxwppvkz.streamlit.app"><b>▶ Live demo</b></a> ·
  <a href="docs/evaluation.md"><b>Evaluation</b></a> ·
  <a href="docs/writeup.md"><b>Writeup</b></a> ·
  <a href="docs/model_card.md"><b>Model card</b></a>
</p>

<br/>

> **A controlled experiment, not a product claim.** Two detectors were trained that differ
> in exactly one thing: how their synthetic training half was produced. Everything else,
> the human corpus, the backbone, the schedule, the seed and the false-positive budget, was
> held fixed. This repository reports what that changed, including where it changed nothing.

<br/>

</div>

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

### In distribution, the two arms are the same detector

| Metric | Arm A · random | Arm B · mirrors |
|---|---|---|
| AUROC | 0.999974 | 0.999971 |
| FNR at the 0.1% FPR budget | 0.719% | **0.430%** |
| Expected calibration error | 0.004467 | **0.003713** |
| Deployed threshold | 0.996956 | 0.992285 |
| Realised FPR | 0.0502% | 0.0502% |

Both are essentially perfect on held-out data from their own distribution, which is exactly
why in-distribution numbers are not evidence of anything.

### Out of distribution, the picture splits three ways

Every arm evaluated on 4,000 documents per benchmark, 2,000 human and 2,000 AI, at the
threshold each arm actually deploys.

| Benchmark | AUROC · A | AUROC · B | Miss rate · A | Miss rate · B | ECE · A | ECE · B |
|---|---|---|---|---|---|---|
| **HC3** | 0.658 | **0.885** | 94.0% | **62.9%** | 0.423 | **0.183** |
| **RAID** | 0.779 | 0.776 | 82.5% | **78.0%** | 0.372 | 0.354 |
| **MAGE** | 0.588 | 0.628 | 96.1% | 86.7% | 0.441 | 0.381 |

### The test that decides it

AUROC compares rankings. A detector is deployed at an operating point. So each arm's
threshold was re-fit to spend the **same** human false-positive budget, and McNemar's test
run on the discordant pairs, because both arms score the *same documents* and their errors
are correlated.

| Benchmark | B catches, A misses | A catches, B misses | χ² | p |
|---|---|---|---|---|
| **HC3** | **119** | 39 | 39.50 | 3.3 × 10⁻¹⁰ |
| **RAID** | **122** | 9 | 95.76 | < 10⁻¹⁵ |
| **MAGE** | 28 | 19 | 1.36 | **0.243** |

**The finding, stated honestly.** On HC3 the mirror arm is better by every measure. On RAID
the two arms have *identical AUROC*, and the sign of that difference reverses in **72.7% of
10,000 paired bootstrap resamples**, yet at a matched budget the mirror arm catches 122
documents the control misses against 9 the other way. Mirroring moved the low-false-positive
tail without moving the ranking. On MAGE it did nothing at all.

### The limitation, in the same breath

> Neither arm is deployable out of distribution. Against unseen generators both miss
> **63% to 96%** of AI text at their deployed threshold, and calibration collapses from
> ECE 0.004 in distribution to **0.18 – 0.44** outside it. A confident score from this
> system is not evidence of a confident model. The live demo will show you this if you
> paste in ChatGPT output, and the page says so rather than hiding it.

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
kernel for something worth 4.47%. See [`docs/jd_coverage.md`](docs/jd_coverage.md) for what
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
[`tests/unit/test_serving_batch_size.py`](tests/unit/test_serving_batch_size.py) reads the
committed artifact and fails if the constant and the measurement ever disagree again.

`forge/inference/batching.py` says its wait window is tuned "against the P95 latency
budget in the release gate". Until this sweep, no P95 had ever been measured, so two
modules were citing a budget that had no artifact behind it. There is one now.

> **Two caveats, before anyone else finds them.** `torch_threads` was 4 on the machine
> that produced this. On a compute-bound CPU forward, thread count is a larger lever than
> batch size, so the absolute figures are probably understated; the batch comparison holds
> because every point ran under the same thread count. And batch 32 got 3 samples under
> the sweep's per-point time budget, so treat the size of its degradation as approximate
> and its direction as real. The artifact records the sample count per row for exactly
> this reason.

**Scope.** This times the model forward over pre-tokenised windows. Tokenisation,
windowing and HTTP are not included, so a real request costs this plus those.


### Spark, and why one machine cannot demonstrate its value

The mining scan is the one job in FORGE shaped for Spark: it reads a reserve pool sized in
millions, runs one independent forward per document, keeps the small fraction the detector
is confidently wrong about, and needs no shuffle, no join and no cross-document state.
`forge/hard_negative/spark_scan.py` is that job.

Run in Spark **local mode** on a 10-core machine, scanning the same 40 documents at every
partition count, with the host's cores divided among the partitions so they do not
oversubscribe:

| Partitions | Threads each | Docs/s (compute) | Speedup | Efficiency | Skew |
|---|---|---|---|---|---|
| 1 | 10 | 4.33 | 1.00 | 100.0% | 1.00 |
| 2 | 5 | 5.67 | 1.31 | 65.5% | 1.03 |
| 4 | 2 | 5.45 | 1.26 | 31.5% | 1.03 |

Total threads is 10 in every row. **The ceiling is about 5.5 docs/s however the cores are
sliced.** On one host, Spark partitioning redistributes cores; it does not add any. Two
partitions of five threads beats one partition of ten, because torch's intra-op scaling is
sublinear, and past that the gains are gone. Skew stays at 1.03, so the partitions are
balanced and the machine is simply full.

> **The measurement I nearly published instead.** The first sweep left torch at its default
> 4 threads regardless of partition count, and reported a 1.72x speedup at four partitions.
> That number was real and meaningless: the serial baseline was using 4 of 10 cores while
> the parallel arms used more. Handicap the baseline and any parallel speedup looks good.
> Fixing it made the headline *worse*, from 1.72x to 1.31x, because the baseline improved
> by 51% and the parallel arms by less. A speedup is a ratio, and a ratio is only as honest
> as its denominator.

**So the Spark claim this repo makes is narrow.** The job exists, is tested, parallelises
without skew, and refuses to mine from anything that is not the reserve pool. It has never
run on a cluster, and `data/reserve/` is empty because the corpus is not redistributed. A
single host cannot show what Spark is for, and a 1.31x local number is not evidence that it
would help at 5M documents. The architecture argument is in
[`docs/jd_coverage.md`](docs/jd_coverage.md); this table is only evidence that the job runs
and scales the way its shape predicts.

Records: [`reports/experiments/profile/`](reports/experiments/profile),
[`reports/experiments/scaling/`](reports/experiments/scaling),
[`reports/experiments/inference/`](reports/experiments/inference) and
[`reports/experiments/spark/`](reports/experiments/spark).

---

## 🖥️ The Interface

Two shells over the same detectors: a FastAPI reference page and a Streamlit page that is
what deploys. Both render the same cards from the same payload against the same stylesheet,
so they cannot drift into disagreeing with each other. The captures below are from the
reference page.

<div align="center">
<img src="images/text_verdict.png" width="92%" alt="Text tab: both arms scored side by side"/>
<br/>
<em>Both arms scored on the same document. The headline is the deployed arm; the other sits
beside it because the comparison IS the experiment. Every number a verdict rests on is on
screen: the threshold, the budget it was fitted at, and that arm's validation FNR and ECE.</em>
</div>

<br/>

<div align="center">
<img src="images/image_verdict.png" width="92%" alt="Image tab: verdict, evidence and file signals"/>
<br/>
<em>Declaration-first verdict logic. The detector's own probability leads the evidence panel,
supporting signals report a word rather than a percentage, and nothing here is combined into
a single invented score.</em>
</div>

<br/>

<div align="center">
<img src="images/image_robustness.png" width="92%" alt="Robustness across eleven transforms"/>
<br/>
<em>Eleven edits an image meets in the wild, each re-scored and compared against the
original. "flipped" means that transform changes the answer. This asks whether the VERDICT
survives redistribution, which is a different question from whether a forensic signal does.</em>
</div>

<br/>

<div align="center">
<img src="images/image_attribution.png" width="92%" alt="Occlusion attribution"/>
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

**918 tests**, and the interesting ones are not unit tests. They are regression tests, each
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
git clone git@github.com:AKilalours/Panagram_Forge.git
cd Panagram_Forge
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
Panagram_Forge/
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
├── docs/                   # evaluation · writeup · model card · data spec
├── demo/                   # held-in AI samples for testing the text tab
└── tests/unit/             # 918 tests, most named after a real bug
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
