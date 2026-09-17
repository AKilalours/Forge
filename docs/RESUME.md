# Where this is, and what is left

Updated 2026-09-17 after the regeneration, retraining and evaluation were finished.

## What happened

Every evaluation artifact in this repository had been produced at commit `d090c9b1`, whose
`VLLMGenerator` sent raw prompts to instruction-tuned models with no chat template. All four
held-in families were being prompted as base models: they continued text instead of following
instructions, a probe measured 34% acceptance with 169 of 565 attempts empty, and the
surviving documents were continuations rather than answers. So the published AUROC and every
out-of-distribution number described a detector trained against base-mode continuations.

Fixed, both arms regenerated as dataset v0.2-min, both retrained, everything re-evaluated.

## The result

Two arms differing only in matching: same roster, same decoding grid, same validation band,
same document budget, length-matched to the human corpus, arm isolation verified against the
prompt version stamped on every document.

| benchmark | arm A random | arm B matched | gap | 95% CI | reversals / 10k | McNemar p |
|---|---|---|---|---|---|---|
| HC3 | 0.7963 | 0.9504 | +0.1541 | 0.1422 to 0.1660 | 0 | 7.3e-61 |
| RAID | 0.7187 | 0.7871 | +0.0685 | 0.0582 to 0.0790 | 0 | 5.3e-73 |
| MAGE | 0.5966 | 0.6124 | +0.0158 | 0.0035 to 0.0279 | 65 | 0.105 |

Matched generation transfers on HC3 and RAID, decisively on both tests. MAGE is a null: the
ranking gap is real but 0.016 and it does not reach the decisions. Both arms sit near chance
there.

**The caveat that travels with all of it.** Even the winning arm misses 62% of AI documents on
HC3 at its deployed threshold, and the control arm misses 99.8% on RAID. Ranking partially
transfers across a domain shift. The operating point does not transfer at all.

## Where the evidence lives

- `Akilalourdes/forge-corpus-v0.2-min` (private dataset): `random/` 19,992 docs, `mirrors/`
  19,736 docs, plus a copy of `reports/`
- `Akilalourdes/forge-detect-weights` (private model): both checkpoints, both run summaries
- `reports/experiments/` in this repo: all artifacts, committed
- W&B: `akilalourdes-student/forge`, runs `ms2oreiz` (arm A) and `97c519m6` (arm B)

## Traps, recorded because each one cost real time

**transformers must be `>=4.40,<5`.** `pip install vllm` pulls 5.17, which makes the
tokenizer return no usable offsets, every token label becomes `ignore_index`, the token loss
reduces over zero elements, and training dies with a NaN that the error message blamed on
fp16 precision. Install the `train` extra without vLLM for training and evaluation; install
vLLM only for generation, and downgrade afterwards.

**The evaluation needs the `data` extra.** `pip install -e ".[dev,serve,train]"` is not
enough; RAID, MAGE and HC3 all load through `datasets`.

**`scripts/eval_all.sh` is idempotent per cell.** It skips any cell whose JSON already
exists, so old artifacts must be deleted before a rerun or it will report "already done" and
leave the previous corpus's numbers in place looking freshly generated.

**The adversarial lab is CPU-only.** `forge.inference.scorer` has no device handling, so
`forge evaluate` never touches the GPU. Run it on a laptop, not a rented card.

**RunPod containers have no volume.** 30 GB of container disk, destroyed on termination.
Export to HuggingFace before stopping, and read it back before believing it.

## Open

1. **Adversarial lab.** `forge evaluate` for both arms. CPU-only, free, nothing waits on it.
2. **`tier1_comparison.json`.** Marked superseded. Its `length_confound_removed` block was
   measured on v0.1-min and needs re-running on v0.2-min with `scripts/length_gate.py`.
3. **Serving weights.** The Streamlit demo pulls from `forge-detect-weights` and the
   checkpoints there are 2.21 GB because they carry optimizer state. Strip a model-only copy,
   around 740 MB, or the free tier cannot load it.
4. **Beam.** Named in the target role, not started. CPU, free, 4 to 6 hours.
5. **CUDA kernel.** The profiler found `aten::scatter_add_` at 20.87% of device time against
   4.47% for all GEMMs combined, three independent measurements agreeing. That is the
   motivated target: measured baseline, named bottleneck, reportable delta. 15 to 25 hours and
   it needs a GPU throughout.
6. **Generation at scale.** 20,000 documents took 27 minutes on one L40S. A 100,000-document
   run plus a multi-GPU sharding measurement costs about $20 and turns an extrapolation into a
   measurement.
7. **Image side.** Still a third-party baseline with an operating point fitted in sample on 29
   images, which the README states. `src/forge/image/` has the full two-arm pipeline written
   and never run, and there is no image config and no CLI command, so it is two to three days
   of work. Nothing in the target role asks for it.
