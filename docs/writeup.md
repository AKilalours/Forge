# FORGE: what failure-driven synthetic data actually bought

**Failure-Driven Synthetic Data Generation for Robust AI-Content Detection**
Tier 1 results, September 2026. Code and every run record: `github.com/AKilalours/forge-ai-detection`.

## The question

Detectors for AI-generated text are trained on synthetic data, and the usual recipe is to
generate a pile of it and hope the pile is representative. The question here is narrower and
testable: **if you generate synthetic documents that mirror specific human documents, does
the resulting detector generalize better to generators it has never seen than one trained on
the same volume of unmatched synthetic text?**

Two arms, identical in everything except which AI documents they contain.

- **Arm A, random synthetic.** 30,000 AI documents generated to length targets drawn from
  the human corpus distribution.
- **Arm B, matched mirrors.** 30,000 AI documents, each one generated from a specific human
  source document and accepted only if it matches that source's length within a band.

Data budget is held equal by construction. Without that, any improvement is "more data" and
says nothing about selection.

Both arms: 20,000 human and 20,000 AI documents, two epochs, DeBERTa-v3-base, four held-in
generator families at 1.7B to 3.8B parameters, one RTX 4090, about 21 minutes per arm. The
operating point throughout is a **0.1% false-positive budget on human text**, because a
detector that accuses innocent writers is worthless regardless of its recall.

## In distribution, the comparison cannot resolve

| | Arm A | Arm B |
|---|---|---|
| Human FPR | 0.000502 | 0.000502 |
| AI FNR | 0.719% | 0.430% |
| AUROC | 0.99997 | 0.99997 |
| ECE | 0.0045 | 0.0037 |

A 40% relative reduction in misses. It means nothing. That gap is fifteen missed documents
against nine, a two-proportion z-test gives **p = 0.217**, and detecting an effect this size
at 80% power would need about 10,761 AI documents per arm in validation against the 2,090
available. The validation split is also saturated: AUROC 0.99997 on both arms, scoring text
from the four families they were trained on. Two nearly perfect models cannot be separated.

The interesting number here is 0.217, not 40%.

## Out of distribution, the headline claim dies and a better one replaces it

Both checkpoints against three benchmarks built from generators neither arm saw: HC3, MAGE
and RAID, 4,000 documents each, balanced.

| Benchmark | Arm | AUROC | FNR at a matched 0.1% FPR |
|---|---|---|---|
| HC3 | A | 0.6583 | 0.9695 |
| HC3 | B | **0.8851** | **0.9295** |
| MAGE | A | 0.5877 | 0.9825 |
| MAGE | B | 0.6279 | 0.9780 |
| RAID | A | **0.7792** | 0.8135 |
| RAID | B | 0.7764 | **0.7570** |

Paired bootstrap over documents, 10,000 resamples, both arms recomputed on each resample
because they score the same documents:

| Benchmark | AUROC(B) - AUROC(A) | 95% CI | Sign reverses in |
|---|---|---|---|
| HC3 | +0.2269 | [0.2125, 0.2413] | 0.0% |
| MAGE | +0.0403 | [0.0290, 0.0513] | 0.0% |
| RAID | **-0.0028** | [-0.0116, 0.0061] | **72.7%** |

**So "mirrors give better AUROC" is not the claim the data supports.** On RAID the arms are
tied with the control marginally ahead. Reporting the HC3 number alone would be the more
flattering half of a true statement, which is its own kind of dishonesty.

But AUROC is the wrong instrument. A detector is deployed at exactly one threshold, far out
in the tail of the human score distribution, and AUROC averages over every threshold
including the thousands nobody will ever use. So: re-fit each arm's threshold on the
benchmark until both spend the **same** two-document false-positive allowance, then count
the AI documents where the arms disagree. McNemar, paired:

| Benchmark | A misses, B catches | A catches, B misses | exact p |
|---|---|---|---|
| HC3 | 119 | 39 | 1.3e-10 |
| MAGE | 28 | 19 | 0.243 |
| RAID | **122** | **9** | 1.9e-26 |

**On RAID the two arms have the same AUROC to within noise, and yet at a usable operating
point the mirror arm catches 122 AI documents the control misses while losing only nine in
the other direction.** Failure-driven mirroring moved the low-false-positive tail without
moving the ranking. That effect is invisible to the metric the field reports, and the tail
is the only region a detector is ever deployed in.

On MAGE there is no effect at all: 28 documents against 19, p = 0.243. Two of three, not
three of three.

## The negative result, which matters more

**Neither arm is deployable out of distribution.** At the threshold a user would actually
get, fit on the in-distribution validation split, both miss **63% to 96%** of AI text,
against 0.43% to 0.72% in distribution.

And calibration does not degrade under shift, it collapses:

| ECE | in distribution | HC3 | MAGE | RAID |
|---|---|---|---|---|
| Arm A | 0.0045 | 0.423 | 0.441 | 0.372 |
| Arm B | 0.0037 | **0.183** | 0.381 | 0.354 |

Two orders of magnitude. The model stays loudly confident while becoming wrong, which is the
worst available failure mode for a product that has to abstain. Arm B is consistently the
less broken of the two and is less than half as miscalibrated on HC3, but "less broken" is
not "usable".

A detector trained on four families at 1.7B to 3.8B parameters does not transfer. No
arm-versus-arm difference changes that.

## What had to be fixed before any of this was worth reporting

The project's recurring failure mode was **checks and configs that reported success while
measuring something other than what their name claimed**. Thirteen instances so far. Four
were mine, in code written for this project. The ones that changed a result:

**A length confound that made the whole thing fake.** A detector could score **AUROC 0.841
without reading a word**, purely from document length. Generation used one `max_new_tokens`
for every document and the models wrote to the cap, so AI text ran systematically long. Two
complete corpora were discarded rather than trained on. Fixed at source with a per-document
token budget, then matched to the human pool: 0.841, then 0.720, then **0.546**. Human median
256 words against AI 257. A gate now blocks training on the assembled data, not on the raw
pools, which is where the first version of the gate was looking.

**An inverted benchmark label.** MAGE's `src` field marks machine text with 0, not 1. The
parser assumed 1, and an existing test had locked the wrong guess in with invented fixtures.
Left alone this would have reported AUROC 0.95 where the truth was 0.05. Settled by measuring
the field over 6,000 rows rather than by flipping until the number looked better.

**A significance test that was measuring the threshold.** The first out-of-distribution test
compared each arm's FNR at *its own* deployed threshold. Those thresholds spend different
false-positive budgets, 0.55% against 2.4% on HC3, so an arm permitted to be four times more
trigger-happy was being credited for missing less. It was also an unpaired test on paired
data. It printed `significant: true` for all three benchmarks. Corrected, **MAGE is not
significant**. That verdict was an artefact.

**Checkpoint selection that never selected.** Selection compared `fpr_at_budget`, which is
measured at a threshold moved until the budget is met and is therefore pinned by
construction. Every epoch of both arms reported 0.000502, one false positive in 1,993
documents, and a `<=` comparison handed the win to the last epoch. `best.pt` was
byte-identical to `last.pt` in both runs. A constant never regresses, which is why nothing
looked wrong for weeks.

Every one of these is committed with the test that catches it. The per-document score arrays
are in the repository, 216 KB, so the bootstrap and the McNemar table both reproduce on a
laptop with no GPU.

## Limits, stated plainly

Four generator families at 1.7B to 3.8B, so absolute FNR looks better than a frontier-scale
roster would give. Roughly 2:1 human to AI in training, identical across arms. A 0.1% budget
over 2,000 human documents is a **two-document** allowance, so the matched thresholds are
coarse. Arm B's mirrors passed a stricter length filter than arm A's documents, which is the
first alternative explanation a reader should propose and cannot be ruled out from two runs.
One GPU, so no scaling claim is made.

## Next

Arms C (hard negatives mined from arm B's own failures) and D (adversarial). The mining,
atlas and selection code is written and tested; the entrypoint that would run it is not
wired, and the arm C config currently points at arm B's data and trains for three epochs
against two, both of which would confound the comparison. Those are fixed before the run,
not after.
