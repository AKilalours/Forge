# Runbook: from zero to a first committed result

> **Historical reproduction guide.** The first training and held-out results are already
> committed. See [`evidence.md`](evidence.md) for current measurements before following
> this sequence.

Goal: one measured baseline in `docs/evaluation.md`. That single number moves the project
from "an implementation" to "an experiment", and it costs about **$2**.

Read `docs/compute_plan.md` section 6 for why the scale is what it is and what the
reduced scale costs you.

---

## Before you spend anything

**Step 0, on your MacBook, free.**

```bash
cd ~/Projects/forge
make setup
make test          # 296 tests must pass
make spec-check
make smoke         # offline ingestion, fixture corpus
make smoke-mirror  # offline mirror engine, fake generator
```

If any of that fails, fix it here. Debugging on a paid pod is the most expensive place to
find a bug that a local test would have caught.

---

## Step 1, ingestion. Free, on your Mac, overnight.

Ingestion needs network, disk and CPU. It does not need a GPU, and paying for one to
stream FineWeb is money burned.

```bash
pip install -e ".[data]"
make min-ingest
```

Two things to know about this step.

**The reserve pool is separate and must be built in the SAME pass.** `make min-ingest`
builds only the training pool, which is all the baseline needs. When you get to Phase 4
mining, use `--with-reserve`:

```bash
forge ingest --config configs/data/human_minimal.yaml --out data/silver \
  --with-reserve --reserve-out data/reserve
```

One pass, one deduplicator, so the two pools are guaranteed disjoint. Running two
separate commands over the same stream produces two OVERLAPPING pools, and a reserve
document that is also a training document has been memorised, so it will never surface as
a false positive and the mining pass finds nothing.

**Watch for a SHORTFALL warning.** The minimal config caps documents at 400 tokens, and
most FineWeb documents are longer than that, so the majority are rejected as
`too_long_tokens`. If the stream runs out before the quota is met the run now says so
instead of quietly under-delivering. The fix is `--read-multiplier 50`, or relaxing
`max_tokens` in the config and accepting multi-window documents.

Produces 60k train + 10k eval + 500k reserve, short documents only, with `MANIFEST.json`.
Expect a **low keep rate**. Web text is dirty. Watch the rejection breakdown:

- keep rate around 5 percent or below means a filter is misconfigured
- keep rate around 95 percent means the filters are not doing anything

Both are bugs. Something in the 15 to 40 percent range is normal.

---

## Step 2, launch a pod

RunPod, **1x RTX 4090** at about $0.74/hr, or **1x A40** at about $0.44/hr.

- Template: RunPod PyTorch 2.x, CUDA 12.x
- Volume: 50 GB network volume mounted at `/workspace`, so data survives a pod restart
- **Do not pick a T4.** No bf16, and DeBERTa-v3 in fp16 is known to produce NaN losses.

```bash
cd /workspace && git clone <your repo> forge && cd forge
bash scripts/runpod_bootstrap.sh
```

The bootstrap refuses to continue on a GPU without bf16 rather than letting you discover
it eight hours into a run.

Upload the ingested Parquet from your Mac, or re-run `make min-ingest` on the pod if the
pod's network is faster than yours.

---

## Step 3, pin the generator revisions

```bash
forge pin-revisions --config configs/generation/generators_minimal.yaml --write
```

Rewrites `TODO_PIN_AT_FIRST_RUN` with real commit shas. Without this the dataset is not
reproducible, and `VLLMGenerator` refuses to construct.

All four held-in families are non-gated. The held-out `gemma` is gated: accept its licence
on huggingface.co and set `HF_TOKEN`, or swap in `Qwen/Qwen2.5-1.5B-Instruct`.

---

## Step 4, generate a 5k sample FIRST. About 10 minutes, $0.15.

```bash
forge mirror --config configs/generation/mirror_minimal.yaml \
  --humans data/silver --out data/silver/mirrors-probe \
  --backend vllm --limit 5000
```

**Read the rejection stats before going further.** They are the early warning:

| Symptom | Meaning | Action |
|---|---|---|
| high `assistant_preamble` | the model ignores the "no preamble" rule | revise the prompt, bump to `mirror_v2` |
| high `too_short` / `too_long` | length control is not working | check `target_tokens` reaches the prompt |
| any `near_duplicate_of_source` | the model is reconstructing the source | attribute extraction is leaking text |
| acceptance below ~70 percent | something systematic is wrong | stop and diagnose |

Fixing a prompt after generating 60k documents means regenerating 60k documents.

---

## Step 5, full generation. About 1.5 hours, $1.10.

```bash
make min-mirror
```

---

## Step 6, smoke the training loop. Two minutes, free.

```bash
make min-smoke     # 20 steps on 200 examples
```

This exists to catch shape errors, tokenizer mismatches, NaN losses and OOM in two
minutes rather than two hours into a real run. **Never skip it.**

---

## Step 7, the run that changes the project. About 1 hour, $0.75.

```bash
export WANDB_API_KEY=...    # optional
make min-train
```

Watch for:
- **loss decreasing** and not NaN
- **`fpr_at_budget` reported each epoch**, because checkpoint selection uses it, not loss
- `examples_per_second`, which is your first real throughput number for `docs/compute_plan.md`

Output lands in `outputs/forge_min_baseline/`: `best.pt`, `last.pt`, `summary.json`.

---

## Step 8, commit the result

```bash
cp outputs/forge_min_baseline/summary.json reports/experiments/
```

Then fill the **Baseline** row of `docs/evaluation.md` with the numbers from
`summary.json`, and commit alongside the `code_commit` and `dataset_version` recorded in
that file.

**That commit is the moment this stops being an implementation.**

---

## What comes next, in value order

1. **Mirror arm.** `configs/training/mirror_minimal.yaml`, same budget. This is the first
   half of the research question: do matched mirrors beat random synthetic data?
2. **External evaluation.** RAID-extra, MAGE, HC3, with the contamination check. This is
   what makes the result credible to someone who did not watch you build it.
3. **Hard-negative arm.** The actual thesis.
4. **Phase 7 on 4x A40**, about $1.50, for the multi-GPU data point.

## If the result is negative

Write it up as a negative result. "Failure-driven selection improved human FPR but did not
improve unseen-generator generalisation" is a real finding, it is more credible than
another positive result, and the repo is built so that claim is checkable.

Do not tune until the number looks good. That is the one failure mode none of the tests
in this repo can catch.
