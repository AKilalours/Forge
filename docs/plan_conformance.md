# Plan conformance audit

> **Historical snapshot, 2026-08-31.** The gap table below predates later implementation
> and measured runs. In particular, evaluation artifacts now exist and token metrics plus
> regime split helpers are present in `src/forge/`; consult the current code, CI, and
> [`evidence.md`](evidence.md) before treating any gap as open.

What the original 40-section plan specified, versus what is actually in this repository.
Produced by inspecting the source tree, not from recollection. Last audited 2026-08-31.

**Summary in one line:** the flow matches, the file layout does not, seven specified
items are not built, seven deliberate deviations were made and recorded, and nothing has
been measured.

---

## 1. Does the system follow the plan's flow?

Yes. Every stage in the plan's final architecture diagram exists as working, tested code:

```
human sources -> ingestion -> data lake -> mirror engine -> training dataset
  -> model -> evaluation lab -> failure atlas -> targeted generation -> retrain
  -> release gate -> API + UI -> monitoring -> verified feedback -> failure atlas
```

The flywheel is closed in code. `src/forge/hard_negative/run.py` runs one full turn:
scan reserve, build atlas, cluster, select across clusters, emit targeted generation
inputs, feed the next training set.

---

## 2. Where the file layout differs, and why

The plan's section 35 specifies a flat `src/ingestion/`, `src/modeling/` tree. The repo
uses `src/forge/...` instead.

**Reason:** a flat `src/` is not an installable Python package. `pip install -e .`
against the plan's layout produces top-level modules named `ingestion`, `training` and
`evaluation`, which collide with other packages in any shared environment. `src/forge/`
gives one namespace, one import root, and a working `pyproject.toml`. This is a
correction, not a drift.

Filenames the plan names, and where that functionality actually lives:

| Plan path | Actual location | Note |
|---|---|---|
| `src/modeling/classifier.py` | `modeling/encoder.py` | both heads live on one shared encoder; splitting them across files would imply two models |
| `src/modeling/token_head.py` | `modeling/encoder.py` | same |
| `src/training/fsdp.py` | `training/distributed.py` | holds FSDP **and** DeepSpeed and Ray behind one interface, which is what makes them comparable |
| `src/training/checkpoint.py` | `training/train.py` | save/load/resume |
| `src/evaluation/adversarial.py` | `adversarial/attacks.py`, `adversarial/lab.py` | grew past one file |
| `src/hard_negative/clustering.py` | `failure_atlas/clustering.py` | clustering belongs to the atlas, which both mining and evaluation consume |
| `docker-compose.yml` (root) | `infra/docker/docker-compose.yml` | grouped with the Dockerfiles |
| `configs/data/synthetic.yaml` | `configs/generation/{generators,mirror}.yaml` | split because generator roster and mirror policy version independently |

Modules the plan did not name but that exist because they were needed:
`common/splits.py`, `common/schemas.py`, `common/config.py`, `generation/assignment.py`,
`generation/attributes.py`, `generation/token_labels.py`, `modeling/alignment.py`,
`evaluation/contamination.py`, `evaluation/benchmarks.py`, `inference/decision.py`,
`training/scaling.py`.

---

## 3. Specified in the plan and NOT built

These are real gaps, listed so nobody has to discover them by reading code.

| # | Plan section | Item | Status | Blocked by |
|---|---|---|---|---|
| G1 | 22 | Token-level metrics: token F1, segment F1, boundary F1 | **not built** | nothing. Buildable and testable today |
| G2 | 16 | Prediction smoothing across windows | **not built** | nothing. Recorded as an open ablation, not a design decision |
| G3 | 13 | Code that CONSTRUCTS the R2 unseen-domain and R4 temporal splits | **not built** | nothing. The regimes are configured and the lab scores them; the splitter is missing |
| G4 | 25 | W&B integration | config keys only, no code | a training run to log |
| G5 | 21 | Ablation runner for models A/B/C/D | **not built** | four trained models |
| G6 | 31 | Research dashboard | **not built** | measured numbers to display |
| G7 | 35 | `notebooks/01..04` | directory empty | data and a model |

G1, G2 and G3 are the honest ones: they need no GPU and no network, and they are simply
not done. G4 through G7 cannot be built against nothing.

---

## 4. Deliberate deviations from the plan, and the reason for each

Each is recorded in `docs/data_spec_v1.md` with a version bump.

| # | Plan said | Repo does | Why |
|---|---|---|---|
| D1 | FineWeb `sample-100BT` | `sample-10BT` | 277 GB is not storable on the dev machine. 10BT gives ~15M documents, already above the plan's own 5-20M target |
| D2 | Evaluate on "RAID test" | `RAID-extra` | RAID-test labels are held out. A self-reported number on it is impossible; it needs a leaderboard submission |
| D3 | RAID as-is | `code`, `Czech`, `German` excluded from the headline | FORGE v1 is an English natural-language detector. Scoring it on code moves the number for reasons unrelated to the research question |
| D4 | (not addressed) | Mined hard negatives split at the **cluster** level | The plan says mined negatives enter training but never says where held-out hard negatives come from. A document-level split puts near-duplicates of one failure on both sides, so the held-out score measures memorisation |
| D5 | (not addressed) | Detector abstains in an uncertain band | At a 0.1% FPR budget, forcing a binary call throws away the protection exactly where the model is least sure |
| D6 | (not addressed) | Attacks measured with AND without preprocessing | `normalize()` defeats several attacks before the model sees them. A single number credits two lines of preprocessing to the model |
| D7 | (not addressed) | Contamination near-dup threshold 0.5, not dedup's 0.8 | The error costs run in opposite directions: a dedup false positive loses training data, a contamination false negative invalidates the result |

Two plan items were also **confirmed correct and kept**, having been tested rather than
assumed: group-aware splitting (section 12) and proportional-across-clusters selection
(section 18). The latter was measured: on a five-mode synthetic failure set at budget 200,
proportional covered 4 of 5 modes and global top-k covered 1 of 5.

---

## 5. What has been measured

Nothing about detection quality.

`docs/evaluation.md` is an empty table by design. There is no trained model, so there is
no FPR, no FNR, no AUROC, no ECE, no OOD number and no adversarial number. Every headline
cell in the plan's section 38 table is still blank.

What HAS been measured is the plumbing: 274 tests, a 309-document offline ingestion run,
302 mirrors, 302 splices and 151 controls, and the selection-policy comparison above.

---

## 6. Honest positioning

The plan's section 40 proposes this framing:

> "I built a smaller, reproducible implementation of the failure-driven training loop and
> investigated whether targeted hard-negative generation improves robustness under
> unseen-generator and adversarial distribution shift."

That framing is **not yet earned**. The word "investigated" claims a result. Today the
accurate version is:

> "I built a reproducible implementation of the failure-driven training loop, with the
> leakage, contamination and measurement-validity failure modes handled explicitly and
> under test. The experiment has not been run."

The second sentence is what makes the first credible. Claiming the investigation before
the numbers exist is the one thing that would undo the care taken everywhere else.
