# Voice Story · the Non-Slop demo

## The idea

**A detector can read patterns. It cannot know why a line matters.** Voice Story lets a
writer choose a passage, explain its meaning in their own words, then export a small,
self-contained keepsake that can be listened to or opened offline. The work that makes it
personal comes from the author; the models are optional evidence and document tools.

This is Forge's authenticity mission made human-facing: help someone preserve the person
behind the words without turning a score into an accusation.

## The three-tool flow

1. **Reducto, by choice:** parse a PDF or DOCX into text blocks with page references. The
   app asks before upload, stages the file in a temporary path, removes that path after the
   request, and keeps the parsed content only in the app session.
2. **Pangram, by choice:** score the text and show returned window-level signals. The app
   disables public dashboard links and never maps a provider score to a claim about who
   wrote the passage.
3. **Hermes, as the reflection companion:** the checked-in `forge-voice-story` skill asks
   one question at a time, protects the author's wording, and refuses to invent memories or
   claim authorship from a detector. Install Hermes with its
   [official guide](https://hermes-agent.nousresearch.com/docs/getting-started/installation),
   copy the skill into `~/.hermes/skills/forge-voice-story/`, then invoke
   `/forge-voice-story` in a new Hermes session.

Forge's matched-mirror arm can also be run locally in the same tab. All external calls are
opt-in. The keepsake is rendered and downloaded in the browser; Forge has no storage route
for it.

## A 90-second walkthrough

1. Open **Voice story**. Start with the fictional sample, or paste a passage you have
   permission to share.
2. Choose one exact sentence. Point to the small remembered detail behind it.
3. Answer the intention and future-reader prompts in your own words.
4. Create the card. Use its **Listen** button, then download the HTML and open it offline.
5. Expand optional signals only if you want to show the analysis path. Explain that they
   help route a review; they do not identify an author.

For a document demo, upload a non-sensitive PDF and explicitly approve Reducto processing.
For a live Pangram demo, set `PANGRAM_API_KEY` in Streamlit secrets and explicitly approve
the text transfer. Do not use a private personal document as the competition sample.

## What the judges can verify

The current status and evidence needed to close the live-provider, Hermes, video,
eligibility, training-scale, generalization, and production-serving gaps are tracked in the
[evidence-backed readiness roadmap](roadmap.md). These are explicit outstanding gates,
not claims that external credentials, event approval, or GPU runs are already available.

| Judge lens | Evidence in the project |
|---|---|
| Product and design | One deliberate user journey, no AI-written memories, source quote and page reference, offline narrated artifact, reduced-motion support, keyboard-visible controls |
| Pangram | Consent-gated SDK adapter, model discovery, bounded window results, no public dashboard link, and explicit no-authorship-claim wording; live credentials still need a real end-to-end smoke test |
| Reducto | Consent-gated parser adapter, typed block/page references, byte cap, temporary local file cleanup, and a fake-client test; a real upload/parse smoke test remains to be captured |
| Nous Research | Versioned Hermes Agent skill at `.hermes/skills/forge-voice-story/SKILL.md`, invoked as `/forge-voice-story` |
| ML engineering | Controlled synthetic-data comparison, reproducible ingestion, transformer training, measured A100/L40S distributed microbenchmarks, model-forward inference experiments, evaluation/release gates, monitoring, and CI |

## Competition timeline and honest disclosure

Forge predates the November 22, 2026 event, and Voice Story was pushed before the event
date. Do not present the repository or Voice Story as if it was built during the hackathon.
Before submitting, ask the organizer whether an existing project with a new event-built
extension is eligible. Identify the pre-existing detector, data, models, and interface;
then list only the work done during the official event window as hackathon work. Preserve
the commit history and disclose AI assistance and third-party assets in the submission.

Suggested organizer question (draft only; not sent):

> Is an extension to my existing open-source AI-content detection project eligible if I
> clearly disclose the pre-existing detector, training data, and app, and build a distinct
> human-centered Voice Story experience during the event? I would use the event APIs only
> with explicit consent and label every pre-event asset.

## Pangram ML Engineer conversation

Show the engineering chain rather than claiming a production scale the evidence does not
support:

- **Data:** adapters, cleaning, deduplication, group-safe splits, provenance, synthetic
  generation, and mirrored versus random controls live under `src/forge/ingestion/`,
  `src/forge/cleaning/`, and `src/forge/generation/`.
- **Training and infrastructure:** `src/forge/training/` contains the fit loop, profiling,
  Ray/DeepSpeed paths, and checkpoint selection. Reports distinguish local experiments
  from cluster-scale operation.
- **Distributed benchmark:** `reports/experiments/scaling/` contains real A100 and L40S
  worker-count measurements. They are short synthetic-batch microbenchmarks; they do not
  represent data-loader throughput or a full distributed training run over the mined corpus.
- **Inference and trust:** `src/forge/inference/` owns windowing, thresholds, and response
  contracts. The image path refuses unsupported or oversized input and labels an untrained
  baseline honestly.
- **Serving measurement:** `scripts/benchmark_serving.py` measures first-request cost and
  HTTP p50/p95/p99 under bounded concurrency. Run it on a local or approved staging service
  before making a production SLO claim.
- **Research:** `docs/evaluation.md` and `reports/experiments/` show the measured result,
  its conditions, the OOD failures, and adversarial gaps.

The strongest explanation is the specific experiment and its limits, not a promise that
one detector catches all AI text. In particular, Forge's own OOD results and failure cases
should be described directly.

## Setup

```bash
pip install -e ".[demo,serve,hackathon]"
streamlit run streamlit_app.py
```

The story and HTML export work without provider keys. The Pangram and Reducto extras enable
their respective opt-in calls. No API key is required for the Hermes skill; Hermes must be
installed separately and configured by the user.
