# Where this is, and exactly what to do next

Written mid-session on 2026-09-17 so the work survives a dropped conversation.
Everything here is state that was verified, not remembered.

## What happened tonight, in one paragraph

Every committed evaluation artifact was produced at commit `d090c9b1`, whose
`VLLMGenerator` sent raw prompts to instruction-tuned models with no chat template. The
models were continuing text rather than following instructions: 34% acceptance, 169 of
565 attempts empty. That means the published AUROC of 0.999974 and every OOD number
describe a detector trained against base-mode continuations. The fix, the regenerated
corpus and the reruns are what the rest of this file is about.

## State as of writing

- repo at `91a203d`, all pushed to `origin/master`
- corpus **v0.2-min** regenerated with the chat template and the words prompt:
  - `data/silver/random` 19,992 docs, `random_v2` only (phi 5055, falcon 5045, qwen 4992, smollm 4900)
  - `data/silver/mirrors` 19,736 docs, `mirror_v2` only (phi 5113, smollm 4971, falcon 4880, qwen 4772)
  - no cross-contamination, checked with pyarrow
- `ai_cap: 19736` in all three arm configs, the smaller arm's count
- AI median 296 words against the human corpus's 257; handled at load time by
  `ai_reference: human`, which resamples the AI pool to the human length histogram
- W&B works: project `akilalourdes-student/forge`
- **transformers must be `>=4.40,<5`**. `pip install vllm` pulls 5.17, which makes token
  labels all `ignore_index`, the token loss reduce over zero elements, and training die
  with a NaN misattributed to fp16. Downgrade AFTER generation, BEFORE training.

## The remaining sequence

On a pod with the corpus present:

```bash
pip install "transformers>=4.40,<5"
forge train --config configs/training/baseline_minimal.yaml 2>&1 | tee -a trainA.log
forge train --config configs/training/mirror_minimal.yaml   2>&1 | tee -a trainB.log
forge evaluate --config configs/training/baseline_minimal.yaml
forge evaluate --config configs/training/mirror_minimal.yaml
```

Then, BEFORE stopping the pod, export. The container disk is 30 GB with no volume behind
it and nothing on it survives termination:

```bash
hf upload Akilalourdes/forge-corpus-v0.2-min data/silver/random  random  --repo-type dataset --private
hf upload Akilalourdes/forge-corpus-v0.2-min data/silver/mirrors mirrors --repo-type dataset --private
hf upload Akilalourdes/forge-detect-weights  outputs             .       --repo-type model   --private
```

Verify by downloading it back and counting files. An upload nobody read back is not a
backup. The old corpus behind the published numbers is gone from every machine, which is
why those numbers cannot be reproduced; do not repeat that.

## Then, in the repo

1. Copy the new run records and eval artifacts into `reports/experiments/`.
2. Update every README table from those artifacts, not from memory.
3. `python scripts/check_readme_claims.py` must pass. It compares each published number
   against its artifact and it is the only reason the tables can be trusted.
4. Add the W&B run URL to the README, which unskips `test_tracking_record.py`.
5. State the two limitations beside the headline figures rather than under them:
   - the generator roster is 1.7B to 3.8B, so absolute FPR and FNR look better than a
     frontier-scale roster would give (this is already written in
     `configs/generation/generators_minimal.yaml` and predates tonight)
   - AI text ran 15% longer at the median than the human corpus before length matching

## Do not ship the current README

Until step 3 passes on the regenerated artifacts, the headline tables describe the old
broken corpus. That is the single most important line in this file.
