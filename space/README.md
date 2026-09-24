---
title: FORGE Detect
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FORGE Detect

AI-content detection, text and images, with the evaluation in the interface.

**Text.** Two detectors trained on an identical data budget, differing only in which AI
documents they contain: one random synthetic, one matched mirrors. Both are shown on every
document, because the comparison between them is the experiment.

**Images.** A published baseline detector, chosen by measuring three candidates against a
labelled probe set, with its polarity and operating point measured rather than assumed. The
attribution panel hides each region and re-scores, so a warm area is one the verdict rested
on.

**Results.** Read from the run records committed in the repository. Nothing on that tab is
recomputed at request time or typed in by hand.

## What this does not do

The text arms were trained on four generator families at 1.7B to 3.8B parameters. Against
unseen generators, measured across HC3, MAGE and RAID, they miss 63% to 96% of AI text and
their calibration error rises from 0.004 to between 0.18 and 0.44. Text from GPT-4-class
models will mostly read as no AI detected. That is the published result, not a malfunction,
and it is on the Results tab.

Source: https://github.com/AKilalours/forge-ai-detection

Built by Akila Lourdes Miriyala Francis
