<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:1a1a4e,100:0d0d2b&height=200&section=header&text=FORGE%20%F0%9F%94%8D&fontSize=58&fontColor=ffffff&fontAlignY=38&desc=Failure-Driven%20Synthetic%20Data%20for%20Robust%20AI-Content%20Detection&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>

### *Built by* **Akila Lourdes Miriyala Francis**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-CPU%20%2B%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/DeBERTa--v3-base-FFB000?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/RunPod-RTX%204090-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Tests-1157%20passing-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/In--distribution%20AUROC-0.99999-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FPR%20budget-0.1%25-0056D2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Headline-Partial%20null%20result-FF8F00?style=for-the-badge"/>
</p>

<p align="center">
  <a href="docs/evaluation.md"><b>Evaluation</b></a> ·
  <a href="docs/writeup.md"><b>Writeup</b></a> ·
</p>

<p align="center">
  <a href="https://panagramforge-cqzwwskdjhbfv6hxwppvkz.streamlit.app"><b>▶ Live demo</b></a>
  <br/>
  <sub><i>Streamlit Community Cloud sleeps when idle. First load takes about two minutes
  while the container wakes and the checkpoint downloads. The evidence above does not.</i></sub>
</p>

<br/>

> **A controlled experiment, not a product claim.** Two detectors were trained that differ
> in exactly one thing: how their synthetic training half was produced. Everything else,
> the human corpus, the backbone, the schedule, the seed and the false-positive budget, was
> held fixed. This repository reports what that changed, including where it changed nothing.

<br/>

</div>

---

**Does synthetic training text that mirrors real human documents beat synthetic text from
random prompts?** Two arms, one variable, three held-out benchmarks. It does, and the
winning model still misses most of the AI documents it meets.

| | AUROC in distribution | AUROC on HC3 | Missed on HC3, at its own threshold |
|---|---|---|---|
| Random prompts | 0.99999 | 0.796 | 87.5% |
| **Matched mirrors** | 0.99989 | **0.950** | **62.2%** |

Ranking transfers. The operating point does not. That gap is the project.

### It runs

<p align="center">
  <img src="images/text_verdict.png" width="90%"
       alt="Text tab: one document scored by both arms, each against its own threshold"/>
  <br/>
  <sub>One document, scored by both arms at once. Each arm carries its own deployed
  threshold, the false-positive budget it was fitted at, and its validation FNR and ECE,
  because a probability without them is decoration. The panel underneath states the
  out-of-distribution miss rate rather than hiding it.</sub>
</p>

<p align="center">
  <img src="images/image_verdict.png" width="90%"
       alt="Image tab: a camera photograph cleared, with the file signals that support it"/>
  <br/>
  <sub>A real photograph. The visual detector returns 0.92% and the file still carries its
  Canon EOS R6 Mark III fields, so the evidence agrees. C2PA and generation markers are
  reported as <b>not found</b> rather than as absence of AI, which is the distinction the
  banner wording turns on: this reads NO AI DETECTED, never HUMAN.</sub>
</p>

<sub>Both captures are from the running app. They are one checkpoint behind the thresholds
quoted elsewhere in this README, so treat the numbers on screen as illustrative of the
layout and the tables in <a href="docs/evaluation.md">docs/evaluation.md</a> as current.</sub>

### Look at it

- **[The evidence page](https://akilalours.github.io/Panagram_Forge/)** — charts, the
  adversarial results, and every infrastructure run, generated from the artifacts so it
  cannot drift from them. Every figure can show the file it came from.
- **[The long-form record](docs/evidence.md)** — the full write-up, including each
  measurement I got wrong first and what the mistake was.
- **[reports/experiments/](reports/experiments)** — the committed JSON everything is
  built from.

### Run it

```bash
pip install -e ".[dev,serve,train,data]"
pytest tests/unit -q                      # the regression suite
python scripts/check_readme_claims.py     # fail if a published number drifted
python scripts/build_evidence_page.py     # rebuild the evidence page
streamlit run streamlit_app.py            # the deployed interface
```

### Scope, stated plainly

Trained on one corpus at one size. Both runners for the mining scan have only run on a
single machine, never a cluster. The adversarial lab finds a case-perturbation attack that
defeats the detector and that preprocessing cannot fix. None of that is hidden in the
long-form record; it is the reason the record exists.

*Akila Lourdes Miriyala Francis*
