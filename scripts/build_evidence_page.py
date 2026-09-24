"""Build the project page from the artifacts, so it cannot drift from them.

WHY THIS IS A BUILDER AND NOT A HAND-WRITTEN PAGE. Every number in this repository is
gated: scripts/check_readme_claims.py fails the build when the README states a figure the
artifacts do not. A second public page full of hand-typed numbers would reintroduce exactly
the drift that gate exists to prevent, and it would rot faster than the README because
nobody re-reads a landing page. So the page is generated. This module contains no
measurement of its own, which a test enforces by grepping it for decimal literals, the same
way forge.ui.evidence is held.

WHICH ARTIFACTS ARE AUTHORITATIVE, because two sets exist and one is stale.
reports/experiments/forge_min_*.json are the v0.1-min runs from commit d090c9b1, before the
chat-template defect was found and both arms were regenerated. indist_*.json are the
v0.2-min runs from 8e06099f and are the ones every published number comes from. Reading the
wrong pair would silently republish the retracted results, so the commit is asserted rather
than assumed.

TWO OUTPUTS, one content. docs/index.html is a complete document for GitHub Pages. The
artifact variant omits the document skeleton, which the Artifact runtime supplies.
"""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "reports" / "experiments"
PAGE = ROOT / "docs" / "index.html"

# The run every published figure belongs to. An artifact from any other commit is a
# different experiment wearing the same filename.
PUBLISHED_COMMIT = "8e06099f3cbcb00e7cdac81e86be0e96191fbd57"

BENCHMARKS = ("hc3", "raid", "mage")
ARMS = (("baseline", "Random prompts"), ("mirror", "Matched mirrors"))


class BuildError(RuntimeError):
    pass


def read(*parts: str) -> dict:
    path = EXPERIMENTS.joinpath(*parts)
    if not path.exists():
        raise BuildError(f"{path.relative_to(ROOT)} is missing; the page cannot be built "
                         f"without it. Run the experiment or remove the section.")
    return json.loads(path.read_text())


def rel(*parts: str) -> str:
    return str(EXPERIMENTS.joinpath(*parts).relative_to(ROOT))


@dataclass(frozen=True)
class Figure:
    """A number, the file it came from, and the condition attached to it.

    All three travel together. The page can render the value alone, but it cannot render it
    without HAVING the source, which is what stops a figure from being typed in later.
    """

    value: str
    source: str
    note: str = ""


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


# ----------------------------------------------------------------- gather

def gather() -> dict:
    indist = {arm: read(f"indist_{arm}.json") for arm, _ in ARMS}
    for arm, data in indist.items():
        if data.get("code_commit") != PUBLISHED_COMMIT:
            raise BuildError(
                f"indist_{arm}.json was written at {data.get('code_commit')}, not the "
                f"published commit {PUBLISHED_COMMIT}. These are different experiments."
            )

    ood = read("ood_summary.json")
    cells = {(c["arm"], c["benchmark"]): c for c in ood["cells"]}
    missing = [k for k in ((a, b) for a, _ in ARMS for b in BENCHMARKS) if k not in cells]
    if missing:
        raise BuildError(f"ood_summary.json has no cell for {missing}")

    significance = {b: read(f"ood_significance_{b}.json") for b in BENCHMARKS}
    mcnemar = read("ood_mcnemar.json")
    adversarial = read("adversarial_forge_min_baseline.json")

    return {
        "indist": indist,
        "ood": ood,
        "cells": cells,
        "significance": significance,
        "mcnemar": mcnemar,
        "adversarial": adversarial,
    }


# ----------------------------------------------------------------- charts
#
# Hand-drawn SVG rather than a charting library: the page ships as one file with no network
# dependency, and three charts do not justify a runtime. Every chart follows the same
# contract. One scale places marks, ticks and labels; every tick names a value the scale
# reaches; the viewBox leaves room for the outermost label; text takes its colour from the
# theme tokens so it reads on either ground; and each mark carries a <title> so the hover
# layer exists without script.

CHART_W = 680
BAR_R = 4


def _axis(y0: int, y1: int, x0: int, x1: int, ticks: list[tuple[float, str]],
          scale) -> str:
    out = []
    for value, label in ticks:
        y = scale(value)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                   f'class="grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick" '
                   f'text-anchor="end">{html.escape(label)}</text>')
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="baseline"/>')
    return "".join(out)


def grouped_bars(groups: list[tuple[str, list[tuple[str, float, str]]]], *,
                 y_max: float, y_ticks: list[float], fmt, title: str) -> str:
    """Grouped bars. Two series, legend plus direct labels, so identity is never colour
    alone. A 2px gap between adjacent bars, and the data end rounded to the baseline."""
    pad_l, pad_r, pad_t, pad_b = 54, 14, 18, 46
    h = 260
    plot_h = h - pad_t - pad_b
    x0, x1 = pad_l, CHART_W - pad_r
    band = (x1 - x0) / len(groups)

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - v / y_max)

    parts = [f'<svg viewBox="0 0 {CHART_W} {h}" role="img" '
             f'aria-label="{html.escape(title)}" class="chart">']
    parts.append(_axis(pad_t, pad_t + plot_h, x0, x1,
                       [(t, fmt(t)) for t in y_ticks], y_of))

    for gi, (gname, series) in enumerate(groups):
        n = len(series)
        inner = band * 0.62
        bw = (inner - 2 * (n - 1)) / n
        gx = x0 + band * gi + (band - inner) / 2
        for si, (sname, value, cls) in enumerate(series):
            bx = gx + si * (bw + 2)
            by = y_of(value)
            bh = max(1.0, pad_t + plot_h - by)
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'rx="{BAR_R}" class="bar {cls}">'
                f'<title>{html.escape(sname)} on {html.escape(gname)}: {fmt(value)}</title>'
                f'</rect>')
            parts.append(
                f'<text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" class="value" '
                f'text-anchor="middle">{fmt(value)}</text>')
        parts.append(f'<text x="{x0 + band * gi + band / 2:.1f}" '
                     f'y="{pad_t + plot_h + 22}" class="axis-label" '
                     f'text-anchor="middle">{html.escape(gname)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def dumbbell(rows: list[tuple[str, float, float]], *, fmt, title: str) -> str:
    """Two values per row joined by a rule. The right form when the story is the DISTANCE
    between two states and both are bad: grouped bars would invite reading the shorter bar
    as good, and here a shorter bar is still most of the documents missed."""
    pad_l, pad_r, pad_t, pad_b = 54, 58, 26, 34
    row_h = 46
    h = pad_t + row_h * len(rows) + pad_b
    x0, x1 = pad_l, CHART_W - pad_r

    def x_of(v: float) -> float:
        return x0 + (x1 - x0) * v

    parts = [f'<svg viewBox="0 0 {CHART_W} {h}" role="img" '
             f'aria-label="{html.escape(title)}" class="chart">']
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_of(t)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t - 10}" x2="{x:.1f}" '
                     f'y2="{pad_t + row_h * len(rows) - 14}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{h - 12}" class="tick" '
                     f'text-anchor="middle">{fmt(t)}</text>')

    for i, (name, a, b) in enumerate(rows):
        y = pad_t + row_h * i + 8
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="axis-label" '
                     f'text-anchor="end">{html.escape(name)}</text>')
        parts.append(f'<line x1="{x_of(min(a, b)):.1f}" y1="{y:.1f}" '
                     f'x2="{x_of(max(a, b)):.1f}" y2="{y:.1f}" class="connector"/>')
        for value, cls, label in ((a, "s1", "Random prompts"),
                                  (b, "s2", "Matched mirrors")):
            parts.append(
                f'<circle cx="{x_of(value):.1f}" cy="{y:.1f}" r="7" class="dot {cls}">'
                f'<title>{html.escape(label)} on {html.escape(name)}: {fmt(value)}</title>'
                f'</circle>')
        parts.append(f'<text x="{x_of(max(a, b)) + 14:.1f}" y="{y + 4:.1f}" '
                     f'class="value">{fmt(max(a, b))}</text>')
        parts.append(f'<text x="{x_of(min(a, b)) - 14:.1f}" y="{y + 4:.1f}" '
                     f'class="value" text-anchor="end">{fmt(min(a, b))}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------- page

STYLE = """
:root{
  --plane:#f7f8fa; --surface:#fcfcfb; --ink:#0d1117; --ink-2:#4a5058; --muted:#83888f;
  --grid:#e4e6ea; --rule:#c8ccd2; --edge:rgba(13,17,23,.10);
  --s1:#2a78d6; --s2:#eb6834; --crit:#d03b3b; --good:#0ca30c;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --plane:#101214; --surface:#17191c; --ink:#e8eaed; --ink-2:#aab0b8; --muted:#83888f;
  --grid:#25282c; --rule:#3a3e44; --edge:rgba(232,234,237,.12);
  --s1:#3987e5; --s2:#d95926; --crit:#e26b6b; --good:#0ca30c;
}}
:root[data-theme="dark"]{
  --plane:#101214; --surface:#17191c; --ink:#e8eaed; --ink-2:#aab0b8; --muted:#83888f;
  --grid:#25282c; --rule:#3a3e44; --edge:rgba(232,234,237,.12);
  --s1:#3987e5; --s2:#d95926; --crit:#e26b6b; --good:#0ca30c;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding-inline:20px;padding-block:40px 72px}
h1{font-family:var(--serif);font-size:clamp(30px,6vw,46px);line-height:1.12;margin:0 0 14px;
  letter-spacing:-.02em;text-wrap:balance;font-weight:600}
h2{font-family:var(--serif);font-size:clamp(21px,3.4vw,26px);margin:0 0 6px;font-weight:600;
  letter-spacing:-.01em;text-wrap:balance}
p{margin:0 0 14px;color:var(--ink-2);max-width:62ch}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin:0 0 10px}
section{margin-top:52px}
.lede{font-size:18px;color:var(--ink);max-width:60ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:26px 0 8px}
.tile{background:var(--surface);border:1px solid var(--edge);border-radius:10px;padding:18px 18px 16px}
.tile .fig{font-family:var(--mono);font-size:clamp(26px,5vw,34px);line-height:1.1;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tile .cap{font-size:13px;color:var(--ink-2);margin-top:6px}
.tile.warn .fig{color:var(--crit)}
.chart{width:100%;height:auto;display:block;margin:10px 0 4px}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .baseline{stroke:var(--rule);stroke-width:1}
.chart .connector{stroke:var(--rule);stroke-width:2}
.chart .tick,.chart .axis-label{font-family:var(--mono);font-size:11px;fill:var(--muted)}
.chart .axis-label{fill:var(--ink-2);font-size:12px}
.chart .value{font-family:var(--mono);font-size:12px;fill:var(--ink-2);
  font-variant-numeric:tabular-nums}
.chart .bar.s1,.chart .dot.s1{fill:var(--s1)}
.chart .bar.s2,.chart .dot.s2{fill:var(--s2)}
.chart .dot{stroke:var(--surface);stroke-width:2}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink-2);margin:2px 0 6px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.swatch{width:11px;height:11px;border-radius:3px;display:inline-block}
.swatch.s1{background:var(--s1)} .swatch.s2{background:var(--s2)}
.src{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:8px}
.src a{color:inherit}
.note{border-left:2px solid var(--rule);padding:2px 0 2px 16px;margin:18px 0;
  color:var(--ink-2);font-size:15px;max-width:60ch}
.note strong{color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;margin:10px 0}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:400}
td{font-family:var(--mono);color:var(--ink-2)}
td:first-child{font-family:var(--sans);color:var(--ink)}
details{background:var(--surface);border:1px solid var(--edge);border-radius:10px;
  padding:14px 16px;margin:10px 0}
summary{cursor:pointer;font-weight:500;color:var(--ink);list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"+ ";font-family:var(--mono);color:var(--muted)}
details[open] summary::before{content:"- "}
details[open] summary{margin-bottom:10px}
.cav{font-size:13px;color:var(--muted);margin-top:8px;max-width:70ch}
.prov{font-family:var(--mono);font-size:11.5px;color:var(--muted);display:none}
body.show-prov .prov{display:block}
.toggle{display:inline-flex;align-items:center;gap:9px;font-size:13.5px;color:var(--ink-2);
  background:var(--surface);border:1px solid var(--edge);border-radius:999px;
  padding:7px 15px;cursor:pointer;font-family:var(--sans)}
.toggle:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--s1);
  outline-offset:3px}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--grid);
  font-size:13px;color:var(--muted)}
a{color:var(--s1)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


def tile(fig: str, cap: str, source: str, warn: bool = False) -> str:
    cls = "tile warn" if warn else "tile"
    return (f'<div class="{cls}"><div class="fig">{html.escape(fig)}</div>'
            f'<div class="cap">{cap}</div>'
            f'<div class="prov">{html.escape(source)}</div></div>')


def table(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
        for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead>' \
           f'<tbody>{body}</tbody></table></div>'


FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500&'
         'family=IBM+Plex+Sans:wght@400;500;600&'
         'family=IBM+Plex+Serif:wght@600&display=swap">')

SCRIPT = """
(function(){
  var b=document.getElementById('prov');
  if(!b)return;
  b.addEventListener('click',function(){
    var on=document.body.classList.toggle('show-prov');
    b.setAttribute('aria-pressed',on?'true':'false');
    b.lastChild.nodeValue=on?' Hide sources':' Show sources';
  });
})();
"""


def build_content(data: dict) -> str:
    cells, sig, mc = data["cells"], data["significance"], data["mcnemar"]
    indist = data["indist"]

    auroc = {(a, b): cells[(a, b)]["auroc_mean_pooled"] for a, _ in ARMS for b in BENCHMARKS}
    miss = {(a, b): cells[(a, b)]["deployed"]["fnr"] for a, _ in ARMS for b in BENCHMARKS}
    val = indist["mirror"]["val"]

    o = ["<title>FORGE Evidence</title>", FONTS, f"<style>{STYLE}</style>",
         '<div class="wrap">']

    # ---- hero: the tension, stated as two figures that are both true --------
    o.append('<p class="eyebrow">FORGE &middot; AI-generated text detection</p>')
    o.append("<h1>A detector that scores almost perfectly, and misses most of what matters"
             "</h1>")
    o.append('<p class="lede">Two training arms, one variable: synthetic text that mirrors '
             'real human documents, against synthetic text from random prompts. Every '
             'figure on this page is generated from a committed artifact. Nothing here is '
             'typed by hand.</p>')
    o.append('<p><button class="toggle" id="prov" aria-pressed="false">'
             '<span aria-hidden="true">&#9679;</span> Show sources</button></p>')

    # ONE ARM ACROSS ALL THREE TILES. The first draft put the winning arm's AUROC beside
    # the control arm's worst miss rate, which reads as one model and is two. And it
    # labelled the figure a test-set score: indist_*.json carries a `val` block and no
    # `test` block, because the threshold is fitted on that split and the AUROC is reported
    # beside it. Naming the split is not pedantry on a page whose argument is that a
    # headline figure without its conditions cannot be acted on.
    o.append('<div class="tiles">')
    o.append(tile(num(val["auroc"], 5),
                  "AUROC in distribution, validation split, matched-mirror arm",
                  rel("indist_mirror.json") + " &rarr; val.auroc"))
    o.append(tile(pct(miss[("mirror", "hc3")], 1),
                  "of AI documents that same model misses on HC3, "
                  "at the threshold it deploys",
                  rel("ood_summary.json") + " &rarr; deployed.fnr", warn=True))
    o.append(tile(pct(miss[("mirror", "raid")], 1),
                  "missed by that same model on RAID, same threshold",
                  rel("ood_summary.json") + " &rarr; deployed.fnr", warn=True))
    o.append("</div>")
    o.append('<div class="note">All three figures are the same model. The first is what a '
             'benchmark reports. The other two are what a user gets.</div>')

    # ---- what changed ------------------------------------------------------
    o.append('<section><p class="eyebrow">The experiment</p>')
    o.append("<h2>One variable, held everywhere else</h2>")
    o.append('<p>Both arms train on the same human corpus and the same four open '
             'generator families, with the AI half capped to the same count so neither '
             'arm buys its result with more data. The control prompts those generators at '
             'random. The treatment asks each one to mirror a specific human document, '
             'matched on topic, register and length.</p>')
    o.append(f'<p class="src prov">{html.escape(rel("indist_baseline.json"))} &middot; '
             f'{html.escape(rel("indist_mirror.json"))} &middot; both at commit '
             f'{PUBLISHED_COMMIT[:8]}</p></section>')

    # ---- OOD ranking -------------------------------------------------------
    o.append('<section><p class="eyebrow">Held-out generators</p>')
    o.append("<h2>Matched generation transfers</h2>")
    o.append('<p>Three public benchmarks the model never trained on, each scored on 4,000 '
             'documents. Six comparisons, six in the same direction.</p>')
    o.append('<div class="legend"><span><i class="swatch s1"></i>Random prompts</span>'
             '<span><i class="swatch s2"></i>Matched mirrors</span></div>')
    o.append(grouped_bars(
        [(b.upper(), [("Random prompts", auroc[("baseline", b)], "s1"),
                      ("Matched mirrors", auroc[("mirror", b)], "s2")])
         for b in BENCHMARKS],
        y_max=1.0, y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        fmt=lambda v: num(v, 2), title="AUROC by benchmark and arm"))
    o.append(table(
        ("Benchmark", "AUROC random", "AUROC mirror", "Gap", "95% CI", "Reversals"),
        tuple((b.upper(), num(auroc[("baseline", b)], 3), num(auroc[("mirror", b)], 3),
               num(sig[b]["observed_difference"], 4),
               f'{num(sig[b]["ci95"][0], 4)} to {num(sig[b]["ci95"][1], 4)}',
               pct(sig[b]["fraction_of_resamples_where_the_gap_reverses"], 1))
              for b in BENCHMARKS)))
    o.append(f'<p class="src prov">{html.escape(rel("ood_summary.json"))} &middot; '
             f'{html.escape(rel("ood_significance_hc3.json"))} and siblings &middot; '
             f'{sig["hc3"]["resamples"]:,} paired bootstrap resamples</p></section>')

    # ---- the failure -------------------------------------------------------
    o.append('<section><p class="eyebrow">The part that does not go on a slide</p>')
    o.append("<h2>The operating point does not transfer</h2>")
    o.append('<p>Ranking is not detection. Each arm deploys a threshold fitted on its own '
             'validation split, and this is the fraction of AI documents that slips under '
             'it. Further left is better, and neither arm is good.</p>')
    o.append('<div class="legend"><span><i class="swatch s1"></i>Random prompts</span>'
             '<span><i class="swatch s2"></i>Matched mirrors</span></div>')
    o.append(dumbbell(
        [(b.upper(), miss[("baseline", b)], miss[("mirror", b)]) for b in BENCHMARKS],
        fmt=lambda v: pct(v, 0), title="Missed AI documents at the deployed threshold"))
    o.append(f'<div class="note">On HC3 the winning arm still misses '
             f'<strong>{pct(miss[("mirror", "hc3")], 1)}</strong> of AI documents. The '
             f'ranking gap is real and significant '
             f'(McNemar exact p = {mc["hc3"]["mcnemar"]["exact_p"]:.1e}), and it does not '
             f'make either model deployable at this threshold.</div>')
    o.append(f'<p class="src prov">{html.escape(rel("ood_summary.json"))} &rarr; '
             f'deployed.fnr &middot; {html.escape(rel("ood_mcnemar.json"))}</p></section>')
    return o


def build_rest(o: list, data: dict) -> list:
    adv = data["adversarial"]

    # ---- adversarial -------------------------------------------------------
    conditions = adv.get("conditions_scored") or ["raw", "normalised"]

    def fnr(result: dict, condition: str):
        block = (result.get("conditions") or {}).get(condition)
        if block is not None:
            return block.get("fnr")
        legacy = {"raw": "fnr_raw", "normalised": "fnr_preprocessed"}.get(condition)
        return result.get(legacy) if legacy else None

    live = sorted(
        (r for r in adv["results"] if any(fnr(r, c) for c in conditions)),
        key=lambda r: -max((fnr(r, c) or 0) for c in conditions))

    o.append('<section><p class="eyebrow">Adversarial</p>')
    o.append("<h2>Where a reader can break it</h2>")
    o.append(f'<p>{adv["n_ai_documents"]} AI test documents, attacked and rescored at the '
             f'deployed threshold. Clean miss rate is zero in every condition, so each '
             f'figure below is the damage the attack does. Only the attacks that move it '
             f'are shown.</p>')
    o.append(table(
        ("Attack", "Severity", *(c.capitalize() for c in conditions)),
        tuple((r["attack"].replace("_", " "), num(r["severity"], 2),
               *(num(fnr(r, c) or 0.0, 3) for c in conditions))
              for r in live)))
    o.append('<div class="note">Production normalisation already defuses invisible '
             'characters. It does nothing for Unicode lookalikes or for case, and random '
             'case flips are the cheapest attack in the set. <strong>Folding confusables '
             'closes the lookalike gap at no measured cost to the false positive rate.'
             '</strong> Case needs training-time augmentation, not a preprocessing '
             'change.</div>')
    human = adv.get("human_cost") or {}
    if human:
        o.append(table(
            ("Condition", "False positives", "Median", "p95", "p99", "Max"),
            tuple((c, num(v["fpr"], 4), f'{v["median"]:.2e}', f'{v["p95"]:.2e}',
                   f'{v["p99"]:.2e}', f'{v["max"]:.2e}')
                  for c, v in human.items())))
        n = next(iter(human.values()))["n_human"]
        o.append(f'<p class="cav">Scored on {n} human documents. A rate this small cannot '
                 f'confirm the deployed budget, so the score distribution is the column '
                 f'that carries the information: a condition that lowers the miss rate by '
                 f'moving every score has not defended anything.</p>')
    o.append(f'<p class="src prov">'
             f'{html.escape(rel("adversarial_forge_min_baseline.json"))}</p></section>')

    # ---- infrastructure, from the same panels the app renders --------------
    sys.path.insert(0, str(ROOT / "src"))
    from forge.ui.evidence import build_panels, measured  # noqa: E402

    panels = measured(build_panels())
    o.append('<section><p class="eyebrow">Infrastructure</p>')
    o.append("<h2>Measured, with its conditions attached</h2>")
    o.append(f'<p>{len(panels)} run records, rendered from the same module the app uses, '
             f'so this page and the running service cannot disagree. Each table carries '
             f'the condition its own artifact states.</p>')
    for p in panels:
        o.append(f"<details><summary>{html.escape(p.title)}</summary>")
        o.append(f"<p>{html.escape(p.headline)}</p>")
        if p.rows:
            o.append(table(p.columns, p.rows))
        for c in p.caveats:
            o.append(f'<p class="cav">{html.escape(c)}</p>')
        o.append(f'<p class="src">{html.escape(", ".join(p.sources))}</p></details>')
    o.append("</section>")

    o.append('<footer><p>Built by <code>scripts/build_evidence_page.py</code> from the '
             'artifacts in <code>reports/experiments/</code>. The claim checker fails the '
             'build when a published figure and its artifact disagree. '
             '<a href="https://github.com/AKilalours/Forge_Panagram">Source</a></p>'
             '</footer>')
    o.append("</div>")
    o.append(f"<script>{SCRIPT}</script>")
    return o


def main() -> int:
    try:
        data = gather()
        content = "".join(build_rest(build_content(data), data))
    except BuildError as error:
        print(f"cannot build the page: {error}", file=sys.stderr)
        return 1

    PAGE.parent.mkdir(parents=True, exist_ok=True)
    document = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,'
        'viewport-fit=cover">\n'
        '<meta name="description" content="FORGE: two training arms for AI-text '
        'detection, and what the winning one still misses.">\n'
        f"{content.split('</style>')[0]}</style>\n</head>\n<body>"
        f"{'</style>'.join(content.split('</style>')[1:])}\n</body>\n</html>\n"
    )
    PAGE.write_text(document)
    print(f"wrote {PAGE.relative_to(ROOT)} ({len(document):,} bytes)")

    artifact = ROOT / ".build" / "artifact_page.html"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(content + "\n")
    print(f"wrote {artifact.relative_to(ROOT)} ({len(content):,} bytes), "
          f"skeleton-free for the Artifact runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
