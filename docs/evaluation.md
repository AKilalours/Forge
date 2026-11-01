# Historical evaluation: v0.1-min

> **Historical snapshot, not the current result.** The tables and conclusions below describe
> the v0.1-min run from 2026-09-03. The current v0.2-min evaluation, artifacts, caveats, and
> benchmark results are maintained in [evidence.md](evidence.md). Use that record for current
> claims; this file is retained to preserve the earlier experiment history.

The ablation table. No cell is filled until the run exists and its `dataset_version`
and `code_commit` are recorded in `reports/experiments/`.

| System | Human FPR | AI FNR | OOD AUROC (HC3 / MAGE / RAID) | Adversarial FNR | ECE |
|---|---|---|---|---|---|
| A: random synthetic | 0.000502 | 0.00719 | 0.658 / 0.588 / 0.779 | not run | 0.00447 |
| B: + synthetic mirrors | 0.000502 | 0.00430 | 0.885 / 0.628 / 0.776 | not run | 0.00371 |
| C: + hard negatives | | | | | |
| D: + adversarial | | | | | |

Arms A and B, `dataset_version` v0.1-min, `code_commit` 4ec8204, 2026-09-03. Full records
in `reports/experiments/`. Both arms: 20,000 human and 20,000 AI documents, two epochs,
DeBERTa-v3-base, one RTX 4090, about 21 minutes each.

## The result, and why it is not yet a finding

At an identical false-positive budget the mirror arm missed fewer AI documents:

| | random | mirror |
|---|---|---|
| False positives | 1 of 1,993 humans | 1 of 1,993 humans |
| Missed AI documents | **15 of 2,086** | **9 of 2,091** |
| FNR | 0.719% | 0.430% |
| Relative reduction | | 40.1% |

**That difference is not statistically significant.** A two-proportion z-test on the FNR
gives z = 1.234, two-sided p = 0.217. Detecting a gap this size at 80% power would need
about **10,761 AI documents per arm in validation**; there are 2,090. The 40% headline rests
on the difference between nine misses and fifteen, and six documents is well inside what
chance produces.

Reported this way deliberately. The number that sells is 40%; the number that is true is
p = 0.217, and the gap between those two sentences is the reason this section exists.

**The validation split is also saturated.** AUROC is 0.99997 on both arms. The detector is
scoring text from the same four generator families it was trained on, and that task is
close to solved at this scale. An in-distribution comparison between two nearly perfect
models cannot separate them.

**So the honest reading is:** the pipeline runs end to end, both arms hit the 0.1% FPR
budget, calibration is good (ECE 0.0045 and 0.0037), and the mirror arm points the right
way without proving anything. The discriminating test is held-out generators, RAID, MAGE
and HC3. That test has now run, and the section below is what it returned.

## Held-out generators: what the OOD run actually returned

Both Tier 1 checkpoints scored against three benchmarks built from generators neither arm
saw in training. Documents are windowed and the windows mean-pooled; the max-pooled variant
is in `reports/experiments/ood_*.json` and is not primary, because max inflates FPR on long
human documents. Full cells in `reports/experiments/ood_summary.json`.

| Benchmark | Arm | AUROC | Deployed FPR | Deployed FNR | FNR at a matched 0.1% FPR |
|---|---|---|---|---|---|
| HC3 | A: random | 0.6583 | 0.0055 | 0.9395 | 0.9695 |
| HC3 | B: mirror | **0.8851** | 0.0240 | 0.6295 | **0.9295** |
| MAGE | A: random | 0.5877 | 0.0085 | 0.9610 | 0.9825 |
| MAGE | B: mirror | **0.6279** | 0.0285 | 0.8665 | **0.9780** |
| RAID | A: random | **0.7792** | 0.0005 | 0.8250 | 0.8135 |
| RAID | B: mirror | 0.7764 | 0.0000 | 0.7805 | **0.7570** |

"Deployed" uses the threshold fit on the in-distribution validation split, which is the only
threshold available at deployment time. The last column refits the threshold on the benchmark
itself to hit 0.1% FPR there. That refit threshold is optimistic and is not available in
production; it is reported only so that the two arms can be compared at the same operating
point rather than at two different ones.

The deployed FPR column is the reason the last column exists. The two arms do **not** sit at
the same operating point out of distribution: on HC3 arm A spends 0.55% of the human budget
and arm B spends 2.4%. Their deployed FNRs are therefore not comparable to each other, and
no test is run on them.

### Significance, and a test that had to be thrown out

Two tests are run, both paired, because both arms score the identical documents.

**AUROC, paired bootstrap, 10,000 resamples.** The resample is over documents, with both
arms recomputed on each one.

| Benchmark | AUROC(B) - AUROC(A) | 95% CI | Resamples where the sign reverses |
|---|---|---|---|
| HC3 | +0.2269 | [0.2125, 0.2413] | 0.0% |
| MAGE | +0.0403 | [0.0290, 0.0513] | 0.0% |
| RAID | **-0.0028** | [-0.0116, 0.0061] | **72.7%** |

**Misses at a matched budget, McNemar.** Each arm's threshold is re-fit on the benchmark so
that both spend the same human false-positive allowance, two documents out of 2,000. The
table counts AI documents where the arms disagree.

| Benchmark | A misses | B misses | A misses, B catches | A catches, B misses | McNemar exact p |
|---|---|---|---|---|---|
| HC3 | 1,939 | 1,859 | **119** | 39 | 1.3e-10 |
| MAGE | 1,965 | 1,956 | 28 | 19 | **0.243** |
| RAID | 1,627 | 1,514 | **122** | 9 | 1.9e-26 |

**A previous version of this section reported the wrong thing.** `ood_table.py` ran a
two-proportion z-test on each arm's FNR at its *own* deployed threshold and wrote
`significant_at_0.05: true` for all three benchmarks. That was wrong twice. The arms sat at
different operating points, 0.55% against 2.4% FPR on HC3, so an arm allowed to be four
times more trigger-happy was being credited for missing less; and the z-test assumes
independent samples when both arms score identical documents. Re-tested properly, **MAGE is
not significant, p = 0.243**, against the p = 0.0 the old test printed. The verdict was an
artefact of the test. `ood_table.py` now emits description and no verdict; adjudication
lives in `scripts/ood_mcnemar.py` and `scripts/ood_significance.py`.

### The reading

**"Mirrors give better AUROC" is not the claim the data supports.** On RAID the arms are
tied, the control arm marginally ahead, and the sign reverses in nearly three quarters of
resamples.

**"Mirrors help at a matched budget on all three benchmarks" is also not the claim.** It
holds on HC3 and RAID, decisively. On MAGE the whole difference is 28 documents against 19,
p = 0.243, which is nine documents of noise.

**What survives is narrower and more interesting than either.** On RAID the two arms have
the same AUROC to within noise, and yet at a 0.1% false-positive budget the mirror arm
catches 122 AI documents the control misses while losing only 9 in the other direction. A
detector is deployed at one threshold in the far tail of the human score distribution, and
AUROC averages over every threshold, so two models can rank equally well overall while
behaving very differently in the only region anyone operates in. **Failure-driven mirroring
moved the low-false-positive tail without moving the ranking.** That is the finding, it
is invisible to AUROC, and it is why the matched-budget test had to exist.

Reporting only the HC3 AUROC would be the more flattering half of a true statement, which is
its own kind of dishonesty.

### Calibration does not degrade under shift, it collapses

| | in distribution | HC3 | MAGE | RAID |
|---|---|---|---|---|
| A: random | 0.0045 | 0.423 | 0.441 | 0.372 |
| B: mirrors | 0.0037 | **0.183** | 0.381 | 0.354 |

ECE rises by two orders of magnitude the moment the generator is unfamiliar. The model does
not merely become less accurate off distribution, it stays loudly confident while becoming
wrong, which is the failure mode that matters most for a product that has to abstain. The
mirror arm is consistently the less broken of the two and is less than half as miscalibrated
on HC3. Neither is usable.

### Neither arm is deployable out of distribution

At the deployed threshold both miss 63% to 96% of AI documents, against 0.43% to 0.72% in
distribution. A detector trained on four families at 1.7B to 3.8B parameters does not
transfer. No arm-versus-arm difference changes that, and closing the gap is what arms C and
D are for.

### Limits of this evaluation

**The budget is two documents.** 0.1% of 2,000 human documents is a two-document allowance,
so every matched threshold sits on the third-highest human score. The resolution is coarse
and a larger benchmark sample would tighten it.

**Window aggregation is barely exercised.** HC3 produced 4,093 windows over 4,000 documents,
so 98% of documents are a single window and mean equals max for them. The claim that max
pooling inflates FPR on long documents is not tested by these runs; it is carried over from
the in-distribution work and stays a stated expectation, not a measured one here.

**"Best checkpoint" meant "last checkpoint" in these runs.** Selection compared
`fpr_at_budget` against the best so far, and that number is measured at a threshold moved
until the budget is met, so it is pinned by construction. Every evaluation of both arms
reported 0.000502, one false positive in 1,993 human documents, and a `<=` comparison then
handed the win to the final epoch. `best.pt` is byte-identical to `last.pt` in both runs.

This does not invalidate the numbers above: the reported metrics were computed on the
checkpoint that was actually saved, and at two epochs the last one is a defensible choice
anyway. What it invalidates is any claim that a checkpoint was *selected*. Selection is now
FNR at the budget, with the budget as a constraint rather than the objective, and
`tests/unit/test_checkpoint_selection.py` fails against the old rule. Not applied to these
runs, which would need retraining.

**The re-fit threshold is not available at deployment.** It is used only to put the arms at
the same operating point. Nothing in the matched-budget table describes what a user would
get.

### Reproducing this without a GPU

The per-document score arrays are committed, 216 KB for all six cells. From a clean
checkout, on a laptop:

```
python scripts/ood_mcnemar.py        # the matched-budget paired test
python scripts/ood_significance.py --benchmark hc3    # the AUROC bootstrap
python scripts/ood_table.py          # the table, which reports and does not adjudicate
```

`ood_mcnemar.py` asserts that both arms scored the same documents in the same order before
computing anything paired, and exits rather than guessing if the arrays are absent. The
tables above were regenerated this way on a machine that never saw the GPU, and every count
matched the original run.

## What made these numbers trustworthy enough to report at all

Before this run, a detector could score **AUROC 0.841 on the control arm without reading a
word of the text**, purely from document length. Generation used one `max_new_tokens` for
every document and the models treated the cap as a target, so AI text ran systematically
long. Two corpora were discarded rather than trained on.

| stage | length-only AUROC (random / mirror) |
|---|---|
| Original corpora | 0.841 / 0.774 |
| Per-document token budget | 0.720 / 0.699 |
| AI length matched to the human corpus | **0.546 / 0.546** |

The final assembled training sets have a human median of 256 words against an AI median of
257. A gate now blocks training whenever length separates the classes by more than 0.15
from chance, measured on the assembled data rather than the raw pools.

Data budget is held equal across A, B and C. Without that, any improvement is just
"more data" and says nothing about failure-driven selection.

## What counts as a result

A positive result: failure-driven mirroring reduces OOD error by a measured amount
using fewer additional synthetic samples than random augmentation, at equal or better
human FPR.

A negative result is still a result and gets written up as one, for example:
"failure-driven selection improved human FPR but did not improve unseen-generator
generalization." What is not acceptable is filling this table with numbers that were
not measured.

## Recorded conditions for the first run (Tier 1, arms A and B)

Written before any result exists, so that the caveats are not chosen after seeing which
way the numbers went.

**Scale.** 60,000 human documents; 30,000 AI documents per arm. Reduced from the planned
60,000 per arm to fit a single-GPU compute budget. The budget is equal across arms, which
is what the comparison depends on; what is lost is statistical precision, not validity.

**Class balance.** Roughly 2:1 human to AI in training. Identical for both arms, so the
comparison is unaffected, but absolute FPR and FNR are shifted relative to a balanced set
and must not be quoted as if they came from one.

**Generator scale.** Four held-in families at 1.7B to 3.8B parameters. Smaller models
produce more detectable text, so absolute FNR will look better than a 7B-to-70B roster
would give. Only the difference between arms is meaningful.

**Validator asymmetry between arms.** The mirror arm accepts a generated document only if
its length ratio to the source falls in [0.6, 1.6]; the random arm uses [0.5, 2.0] against
a target length drawn from the corpus distribution. The thresholds differ because matching
is the point of a mirror and there is nothing to match in the control. The consequence is
that arm B's data passed a stricter filter than arm A's. If arm B wins, this is the first
alternative explanation a reader will propose, and it cannot be ruled out from these two
runs alone. Ruling it out needs an arm B variant generated under the looser thresholds.

**Length overshoot dominated mirror rejection, and it was structural.** The first real
run accepted 18,856 of 30,000 mirrors, a 31.5% rate over 59,920 attempts. The reasons:

| reason | count |
|---|---|
| too_long | 33,912 |
| too_short | 6,070 |
| assistant_preamble | 999 |
| empty | 83 |

Length overshoot is 85% of all rejections and the cause is structural rather than random:
generation used one `max_new_tokens` of 640 for every document, regardless of its target
length. A source with a 250-token target could never be matched, because the model writes
to the cap and lands at a ratio of about 2.6 against a ceiling of 1.6. That also explains
why retries recovered so little: the seed changes between attempts, the length ceiling does
not. Attempt 1 recovered 632 of 3,055 failures on the first family.

Consequence, and the reason the arms are capped by distribution rather than by count: the
mirrors that survive skew long, while the control arm draws its target lengths from the
whole human corpus and does not skew. Capping both arms to an equal count alone would leave
them differing in length distribution as well as in matching, and a detector can learn
length. Both arms are therefore capped with `cap_documents_matching`, which selects the
control arm's documents so its length profile tracks the mirror arm's.

Fix for the next run, not applied here: set `max_new_tokens` per document from its target
rather than globally. Applying it now would mean regenerating both arms.

**Selection pressure from rejection.** First-pass rejection on the mirror arm ran near 40%
for the first generator family. Documents are retried up to twice with different seeds, so
final acceptance is higher, but the accepted set is selected rather than sampled: it
over-represents source documents whose mirrors happened to land in the accepted length
band. The per-reason breakdown is recorded in `reports/experiments/`.

**Single GPU.** One RTX 4090. The FSDP and DeepSpeed paths exist and are tested but were
not exercised for these runs, so no multi-GPU scaling claim is made.

**Not measured here.** Arms C and D, external benchmarks (RAID, MAGE, HC3), the adversarial
suite, and calibration under distribution shift. Those are Tier 2 and their rows stay empty
until they are run.
