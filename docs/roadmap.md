# Evidence-backed readiness gates

This roadmap tracks what remains before Forge can make stronger product, research, and
infrastructure claims. A feature is complete only when its acceptance evidence exists in
the repository. Missing credentials, event approval, or hardware are explicit blockers;
they are not replaced with mocks or estimated results.

## Current evidence

| Area | Verified today | Scope boundary |
|---|---|---|
| Provider adapters | Pangram and Reducto adapters, explicit-consent UI, fake-client tests; a consented fictional-data smoke is recorded in `reports/experiments/providers/live-smoke-20261101T223239Z.json` | Reducto parsed one page in 3,167.84 ms; Pangram returned `ProviderError`. The report contains no content or secrets; hosted Pangram remains disconnected |
| Hermes | Versioned skill installed into Hermes Agent v0.22.5; a local Ollama-backed session was attempted | The local model did not return a completed response; skill behavior is not yet demonstrated. Details: [`hermes_local_smoke.md`](hermes_local_smoke.md) |
| Demo media | 14-second Text/Image capture; six-step tour links to the live Voice Story flow | Video predates Voice Story; the Voice Story tour art is an illustration |
| Event eligibility | Existing project and pre-event Voice Story dates documented; organizer question drafted in `docs/hackathon_story.md` | No organizer response has been received |
| Data scale | 5,638,371 Common Crawl documents streamed/cleaned on one laptop; 90,948 entered the reserve pool | One host, 4 Spark workers, 270/284 segments; not model training at that scale |
| Distributed training | FSDP A100 and FSDP/DeepSpeed L40S worker-count reports | Short synthetic-batch microbenchmarks; not end-to-end training through the mined-data loader |
| Generalization | HC3, RAID, and MAGE evaluation artifacts | Current miss rates remain 62.2%, 79.5%, and 89.1% respectively; these are known failures, not resolved targets |
| Serving | Local FastAPI benchmark: 20 requests at concurrency 1/2/4, schema-checked, zero HTTP errors | One Mac CPU run; p99 has only 20 observations per level and is not a production SLO |

## Closure gates

### 1. Run consented Pangram and Reducto smoke tests

**Status:** blocked on deployment credentials and an explicitly approved test payload.

1. Add `PANGRAM_API_KEY` and `REDUCTO_API_KEY` through the deployment's secret manager or
   Streamlit secrets. Never commit them, paste them into chat, or put them in the README.
2. From a trusted local terminal with those keys set privately, run
   `python scripts/smoke_providers.py --provider both --confirm-external` and type
   `SEND FICTIONAL SAMPLE` at the prompt. The harness sends only its built-in fictional
   passage and generated one-page fictional PDF, then stores sanitized metadata under
   `reports/experiments/providers/`. Select `--provider pangram` or `--provider reducto`
   to test one integration. The harness makes no request if the flag, keys, or typed
   confirmation are missing. Review the JSON before committing it.
3. Use the corresponding app consent controls before each external transfer in the live UI.
4. Save redacted run metadata: provider, timestamp, model/version if returned, status,
   latency, and response schema. Do not save the passage, PDF bytes, or secret values.
5. Verify the app renders provider signals as optional evidence and makes no authorship
   claim. Verify the keepsake flow works after declining provider processing too.

**Exit:** both live requests succeed from the deployed app, consent is visible, and
redacted, reproducible evidence is committed. A fake-client test is not an exit artifact.

### 2. Demonstrate the Hermes skill in a real session

**Status:** Hermes Agent v0.22.5 is installed and the Forge skill is registered. A local-only
Ollama session was attempted, but no completed assistant response was produced, so this gate
remains open.

1. Use a supported local 64K model configuration or a no-cost authorized provider; do not
   make paid API calls just to close this demo gate.
2. Open a fresh Hermes session, invoke `/forge-voice-story`, and use only Forge's fictional
   passage with external providers disabled.
3. Confirm it asks one question at a time, preserves the chosen excerpt, invents no
   personal detail, and asks before external transfer or assembly.
4. Capture a short redacted transcript or session recording. Label the runtime/version and
   model. Record any local runtime limitations rather than presenting a failed attempt as a
   completed demonstration.

**Exit:** one real session is reproducible, the skill is loaded by Hermes, and the output
passes the author-control checks in the skill itself.

### 3. Record the current product flow

**Status:** existing video is Text/Image only; tour's Voice Story screen is explicitly
illustrative.

Record a fresh 30–60 second capture of the live Voice Story UI: select the fictional sample,
choose a source excerpt, complete the writer-authored reflection prompts, decline optional
providers unless the live smoke tests above are complete, create the keepsake, use Listen,
and download/open the offline HTML. Show consent controls without exposing secrets or any
private document. Keep the current Text/Image video separately and label its scope and date.

**Exit:** the MP4 is a real current UI capture, plays in the README/tour, and its chapter
labels accurately describe what is shown. Do not add detector scores unless they are from a
current artifact and visibly labeled with that artifact's date.

### 4. Get written event eligibility guidance and disclose prior work

**Status:** a question is drafted in `docs/hackathon_story.md`; it has not been sent.

The participant should send the draft to the event organizer and retain the written reply.
Disclose that Forge, its detector, training data, models, UI, and Voice Story existed before
the event. If extensions are eligible, define the exact new work to build in the event
window, preserve commit timestamps, and disclose AI assistance and third-party assets.

**Exit:** written organizer guidance is saved or referenced, and the submission clearly
separates pre-existing work from work completed during the event. Do not represent the
existing repository as a hackathon-built project.

### 5. Measure end-to-end training at larger scale

**Status:** multi-million-document ingestion and GPU step microbenchmarks exist; the local
machine has neither CUDA nor MPS, and no RunPod credential is configured.

On the provisioned GPU host, run the repository's pinned, reproducible training pipeline
with the real data loader and a declared corpus size. Start with a small smoke run, then a
pre-registered larger run. Record data/corpus fingerprints, examples and tokens processed,
data-loader wait, optimizer throughput, GPU utilization, peak memory, wall time, cost,
world size, and failed/retried steps. Compare single- and multi-GPU runs at identical global
batch and precision. Keep synthetic-input microbenchmarks in a separate table.

**Exit:** the end-to-end run is reproducible from its config and corpus manifest, has
complete hardware/cost metadata, and its report distinguishes observed values from
projections. Never claim multi-node performance from a single host.

### 6. Improve and remeasure out-of-domain detection

**Status:** baseline failures are measured; a generation token-budget correction is in the
code, but the historical models predate it and have not been regenerated under that
protocol.

1. Regenerate both experimental arms with identical corrected budgets and data protocol.
2. Add source-, domain-, and generator-held-out slices with group-safe splits and a
   contamination check before training.
3. Mine failure cases from training/validation only; keep benchmark test rows untouched.
4. Refit thresholds on validation at the declared human false-positive budget. Report FNR,
   FPR, AUROC, calibration, confidence intervals, and per-slice results.
5. Use paired comparisons and release gates. Report regressions and null results; do not
   select a checkpoint against the test set.

**Exit:** a committed run manifest, checkpoints, evaluation outputs, and claim-checkable
report demonstrate whether the miss rate improved without exceeding the false-positive
budget. No numerical improvement is assumed in advance.

### 7. Establish a production serving profile

**Status:** local CPU baseline exists at 20 requests per concurrency level. The benchmark
harness now permits up to 5,000 samples and caller-selected concurrency levels; it defaults
to localhost and refuses remote hosts unless `--allow-remote` is explicitly set.

For a production-like, authorized target, first run 1,000 requests per point with
`--concurrency 1,2,4,8,16` and a fixed fictional passage. Repeat after warmup and across
cold starts. Capture p50/p95/p99, throughput, status/error rates, service and client
hardware, peak memory, startup time, model load time, and cost per 1,000 requests. Set
explicit timeouts and stop conditions; do not load-test a public production service without
approval.

Example against an approved staging origin:

```bash
python scripts/benchmark_serving.py \
  --base-url https://staging.example.invalid \
  --allow-remote \
  --samples 1000 \
  --concurrency 1,2,4,8,16 \
  --out reports/experiments/serving/staging-load.json
```

**Exit:** repeated staging results meet explicitly chosen latency/error SLOs with complete
hardware and cost context. The existing 20-request Mac sample is a smoke baseline, not an
SLO result.

## Priority order

1. Obtain organizer eligibility guidance and agree on disclosure before submitting.
2. Configure provider secrets, run consented fictional-data smokes, invoke Hermes, and
   capture the real Voice Story flow.
3. Run larger data-backed training and OOD evaluation on provisioned GPU compute.
4. Profile the deployed serving path under an approved staging load, then set SLOs from
   repeated measurements.

External credentials, organizer decisions, GPU capacity, and deployment permissions are
required inputs. This roadmap makes those dependencies actionable; it does not imply that
they have already been completed.
