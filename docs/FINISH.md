# Finish line

> **Historical runbook, superseded.** This status predates the v0.2-min training and
> evaluation artifacts. The current measured results are in [`evidence.md`](evidence.md);
> use the steps below only as a historical reproduction
> recipe, not as the current project status.

State at the time of this runbook: corpus complete at 60,000 documents, with 296 tests green;
training and evaluation had not yet run.

This is the shortest path to a completed project. Do not deviate from the order.

---

## Definition of done

Three tiers. **Aim for Tier 2.** Tier 3 is polish.

| Tier | What it proves | Arms | Cost | Verdict |
|---|---|---|---|---|
| **1** | Do matched mirrors beat random synthetic data? | A, B | ~$3 | a real finding on its own |
| **2** | **Does failure-driven mining beat both?** | A, B, C + RAID/MAGE | ~$8 | **the actual thesis** |
| 3 | Complete plan | + D, multi-GPU, serving | ~$20 | optional |

---

## Step 0 — GitHub. Five minutes, free.

There is no remote yet, and the pod clones from GitHub.

```bash
gh repo create forge --private --source=. --push
```

## Step 1 — Pod. $0.74/hr on a 4090, $0.44/hr on an A40.

Not a T4: no bf16, and DeBERTa-v3 NaNs in fp16.

```bash
cd /workspace && git clone <your repo> forge && cd forge
bash scripts/runpod_bootstrap.sh
forge pin-revisions --config configs/generation/generators_minimal.yaml --write
```

Upload `data/silver/` from your Mac, or re-run `make min-ingest` on the pod. Your local
run spent 97 percent of its hour on the HuggingFace download, not on compute, and the
pod's network is far faster.

## Step 2 — Probe BOTH arms. 15 minutes, ~$0.20. Do not skip.

```bash
forge mirror --config configs/generation/mirror_minimal.yaml \
  --humans data/silver --out data/silver/mirrors-probe --backend vllm --limit 2000

forge generate-random --n 2000 --humans data/silver \
  --out data/silver/random-probe --backend vllm
```

Read the rejection stats for both. Acceptance below 70 percent on either arm means stop
and diagnose. Fixing a prompt after 60k documents means regenerating 60k documents.

## Step 3 — Generate both arms at equal budget. ~2 hours, ~$1.50.

```bash
forge mirror --config configs/generation/mirror_minimal.yaml \
  --humans data/silver --out data/silver/mirrors --backend vllm

forge generate-random --n 60000 --humans data/silver \
  --out data/silver/random --backend vllm
```

**The counts must match.** Equal budget per arm is what makes the comparison about
matching rather than about volume.

## Step 4 — Smoke the trainer. 2 minutes, free. Never skip.

```bash
make min-smoke
```

The trainer now refuses to start unless the config declares `data.arm`, and it verifies
the loaded AI documents actually belong to that arm by checking their recorded
`prompt_version`. If Arm A is pointed at the mirror directory it fails loudly instead of
training on the wrong data and reporting a plausible number.

## Step 5 — Train both arms. ~2 hours, ~$1.60.

```bash
forge train --config configs/training/baseline_minimal.yaml   # Arm A, random
forge train --config configs/training/mirror_minimal.yaml     # Arm B, mirrors
```

## Step 6 — Commit the first result. This is the moment the project changes.

```bash
cp outputs/*/summary.json reports/experiments/
```

Fill rows A and B of `docs/evaluation.md` from those files, with the `code_commit` and
`dataset_version` each records. Commit.

**Tier 1 complete. You now have a measured finding.**

---

## Step 7 — Tier 2: the actual thesis. ~$4 more.

```bash
forge ingest --config configs/data/human_minimal.yaml --out data/silver \
  --with-reserve --reserve-out data/reserve

forge mine --config configs/training/hard_negative_minimal.yaml \
  --reserve data/reserve --model outputs/forge_min_mirror/best.pt

forge train --config configs/training/hard_negative_minimal.yaml   # Arm C
forge evaluate --config configs/eval/regimes.yaml                  # RAID, MAGE, HC3
```

The contamination check must pass before any external number is reported.

---

## Step 8 — The writeup. Free, and it is the actual deliverable.

About 1,200 words:

1. The question, one sentence.
2. What you built, one paragraph. State the scale plainly: 60k documents, 1.7-3.8B
   generators. Never imply frontier scale.
3. The table.
4. **The failure modes you had to defend against.** This is the most interesting section
   and almost nobody else's writeup has one: group leakage, a contamination threshold
   tuned for the opposite error cost, preprocessing credited to the model, a config that
   never reached the cleaner.
5. What you did not measure, and why.

**If the result is negative, write it up as negative.** "Failure-driven selection improved
human FPR but did not improve unseen-generator generalisation" is a real finding, more
credible than another positive result, and this repo is built so the claim is checkable.

Do not tune until the number looks good. That is the one failure mode no test here can
catch.

---

## Then, and only then

Email Pangram with **one specific finding** from section 4. Not the repo link. The repo
goes in the last line.
