# Runbook: the 2026-09-02 run, from generation to a filled results table

The state this assumes: 60,000 human documents ingested to `data/silver`, generation of
both arms running under tmux session `gen` on the pod, writing `/workspace/gen.log`.

Everything here is on the pod unless it says otherwise. Never attach to the tmux session;
read the log instead, because a stray Ctrl-C in an attached pane kills the job.

---

## 1. Wait for generation

```bash
grep "^\[mirror\]" /workspace/gen.log | tail -4     # arm B, mirrors
grep "^\[random\]" /workspace/gen.log | tail -4     # arm A, starts after arm B
```

Done when both stats blocks have printed:

```bash
grep -E "^generated|^  (accepted|attempts|acceptance_rate|rejected|families_used|arm)" /workspace/gen.log
```

**Read the acceptance rate before going further.** Below about 0.6 on either arm means the
generated data is heavily selected and the run needs discussing before it is trained on.

Confirm both arms actually wrote to their own directories, since the two-arms-one-directory
bug is the most expensive mistake available here:

```bash
du -sh data/silver/mirrors data/silver/random
python - <<'PY'
import glob, pyarrow.parquet as pq
for root in ("data/silver/mirrors", "data/silver/random"):
    versions, n = set(), 0
    for f in glob.glob(f"{root}/split=*/*.parquet"):
        t = pq.read_table(f, columns=["mirror", "split"]).to_pylist()
        n += len(t)
        versions |= {r["mirror"]["prompt_version"] for r in t}
    print(f"{root}: {n} docs, prompt_versions={sorted(versions)}")
PY
```

`data/silver/mirrors` must report `mirror_v2` only, `data/silver/random` must report
`random_v2` only. If either shows both, stop: the arms are contaminated and any comparison
between them is meaningless.

---

## 1b. Set the equal budget. Do not skip this.

Each arm's validator rejects at its own rate, so the two arms finish generation with
DIFFERENT accepted counts. Training on those counts as-is confounds the experiment: arm A
winning could just mean arm A had more documents.

Count what each arm actually produced:

```bash
python - <<'PY'
import glob, pyarrow.parquet as pq
for root in ("data/silver/mirrors", "data/silver/random"):
    n = sum(pq.read_metadata(f).num_rows for f in glob.glob(f"{root}/split=*/*.parquet"))
    print(root, n)
PY
```

Take the SMALLER number and write it as `data.ai_cap` in BOTH
`configs/training/baseline_minimal.yaml` and `configs/training/mirror_minimal.yaml`.
The same value in both files. Commit the change, so the table's numbers can be traced to
the budget they were trained under.

`cap_documents` then selects that many documents per arm deterministically, keeping each
arm's split proportions, so both arms train on an equal AI budget without regenerating
anything.

---

## 2. Smoke test. Five minutes, catches config errors before a paid run.

```bash
cd /workspace/forge && git pull && python -m pytest -q 2>&1 | tail -3
make min-smoke
```

`git pull` is safe now and not before: generation has finished, so no running process can
pick up half-changed code.

The smoke run must report a loss AND non-trivial counts for both classes. If it reports
zero AI examples, the limit bug has regressed.

---

## 3. Train arm A, the control.

```bash
tmux new -s trainA -d 'cd /workspace/forge && python -m forge.cli train \
  --config configs/training/baseline_minimal.yaml 2>&1 | tee /workspace/trainA.log'
```

Watch: `tail -f /workspace/trainA.log`. About 40 minutes.

Arm A must load `data/silver/random`. The config's `data.arm: random` is checked against
the `prompt_version` on every document, so a mismatch raises `ArmMismatch` rather than
silently training on the wrong arm.

---

## 4. Train arm B, the mirrors.

Only after arm A finishes; they share one GPU.

```bash
nvidia-smi --query-gpu=memory.used --format=csv    # must be near zero first
tmux new -s trainB -d 'cd /workspace/forge && python -m forge.cli train \
  --config configs/training/mirror_minimal.yaml 2>&1 | tee /workspace/trainB.log'
```

---

## 5. Record the result

```bash
ls outputs/*/summary.json
cat outputs/*/summary.json
```

Copy the numbers into the arm A and arm B rows of `docs/evaluation.md`. Commit the
`summary.json` files under `reports/experiments/` alongside the code commit they came from.

The conditions section of `docs/evaluation.md` was written before any of these numbers
existed. Do not soften it now that they do.

---

## What can still go wrong, and what it means

**`ArmMismatch`** — a config points at the wrong AI directory. Do not "fix" it by relaxing
the check; the check is the only thing standing between you and a false negative.

**"no training windows after windowing"** — the loaded subset contains no `train` split.
Means the (source, split) bucketing regressed.

**"Free memory on device"** — something else still holds the GPU. `tmux ls`, then
`nvidia-smi`. The preflight check should now name this before the model loads.

**Loss goes to NaN** — check `precision` in the config against the GPU. bf16 requires
Ampere or newer; the config refuses fp16 on DeBERTa-v3 for exactly this reason.
