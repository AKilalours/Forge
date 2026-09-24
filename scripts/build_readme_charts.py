"""Draw the README's charts as standalone SVG files, straight from the committed artifacts.

    python scripts/build_readme_charts.py

WHY THIS EXISTS RATHER THAN A SCREENSHOT. Every figure in the README should be derivable
from a file in reports/experiments/. A screenshot of a chart is a claim with no audit trail:
it cannot be regenerated, it does not change when the numbers do, and nothing fails when it
goes stale. These SVGs are rebuilt from the same JSON the tables are gated against, so a
figure that disagrees with an artifact is a bug with a reproduction rather than a picture
nobody can check.

WHY PRESENTATION ATTRIBUTES AND NOT CSS. GitHub sanitises SVG served through a README and
strips <style> elements, so a chart styled by class renders as unstyled black shapes on
github.com while looking correct locally. Every colour here is therefore an attribute on the
element that uses it. The palette is picked to read on both the light and dark GitHub
themes, because a README has no control over which one the reader is using: the greys are
GitHub's own muted tokens, which are contrast-tested against both grounds.

The evidence page draws its own charts from these same artifacts with the page's CSS. The
duplication is deliberate: that page controls its stylesheet and this one does not.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "reports" / "experiments"
OUT = REPO / "images"

W = 720
INK = "#57606a"        # readable on both GitHub themes
MUTED = "#8b949e"
S1 = "#3b82f6"
S2 = "#f97316"
CRIT = "#dc2626"
FONT = "ui-monospace,SFMono-Regular,Menlo,monospace"


def _text(x, y, s, *, size=12, fill=INK, anchor="middle", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{html.escape(str(s))}</text>')


def _frame(title, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" '
            f'width="{W}" height="{h}" role="img" aria-label="{html.escape(title)}">')


def grouped_bars(groups, *, series_names, y_max, y_ticks, fmt, title, colors,
                 note=None):
    """groups: [(label, [v1, v2]), ...]. Bars carry their own value, so the chart is
    readable without hovering and without trusting the reader to interpolate an axis."""
    # pad_t has to clear the legend AND the value label sitting above the tallest bar.
    # At 44 the held-in AUROC label landed on top of the legend text.
    pad_l, pad_r, pad_t, pad_b = 58, 18, 62, 58
    h = 318 + (18 if note else 0)
    plot_h = h - pad_t - pad_b - (18 if note else 0)
    x0, x1 = pad_l, W - pad_r
    band = (x1 - x0) / len(groups)

    def y_of(v):
        return pad_t + plot_h * (1 - v / y_max)

    p = [_frame(title, h)]
    p.append(_text(x0, 20, title, size=13, anchor="start", weight="600"))

    for i, name in enumerate(series_names):
        lx = x0 + 250 * i
        p.append(f'<rect x="{lx}" y="{30}" width="10" height="10" rx="2" '
                 f'fill="{colors[i]}"/>')
        p.append(_text(lx + 16, 39, name, size=11, anchor="start", fill=MUTED))

    for t in y_ticks:
        y = y_of(t)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                 f'stroke="{MUTED}" stroke-width="1" opacity="0.22"/>')
        p.append(_text(x0 - 8, y + 4, fmt(t), size=11, anchor="end", fill=MUTED))
    p.append(f'<line x1="{x0}" y1="{pad_t + plot_h}" x2="{x1}" '
             f'y2="{pad_t + plot_h}" stroke="{MUTED}" stroke-width="1.5"/>')

    for gi, (gname, values) in enumerate(groups):
        n = len(values)
        inner = band * 0.60
        bw = (inner - 3 * (n - 1)) / n
        gx = x0 + band * gi + (band - inner) / 2
        for si, value in enumerate(values):
            bx = gx + si * (bw + 3)
            by = y_of(value)
            p.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                     f'height="{max(1.0, pad_t + plot_h - by):.1f}" rx="3" '
                     f'fill="{colors[si]}">'
                     f'<title>{html.escape(series_names[si])} on '
                     f'{html.escape(gname)}: {fmt(value)}</title></rect>')
            p.append(_text(bx + bw / 2, by - 7, fmt(value), size=11,
                           fill=colors[si], weight="600"))
        for li, line in enumerate(gname.split("\n")):
            p.append(_text(x0 + band * gi + band / 2, pad_t + plot_h + 22 + li * 15,
                           line, size=12))
    if note:
        p.append(_text(x0, h - 10, note, size=11, anchor="start", fill=MUTED))
    p.append("</svg>")
    return "".join(p)


def write(name, svg):
    path = OUT / name
    path.write_text(svg + "\n")
    print(f"wrote {path.relative_to(REPO)} ({len(svg):,} bytes)")


def chart_generalisation():
    """The project's headline, in one picture: ranking survives, the threshold does not."""
    cells = {(c["arm"], c["benchmark"]): c
             for c in json.loads((ART / "ood_summary.json").read_text())["cells"]}
    val = json.loads((ART / "indist_mirror.json").read_text())["val"]

    groups = [("Held-in\n(its own test set)", [val["auroc"], val["fnr"]])]
    for bench, label in (("hc3", "HC3"), ("raid", "RAID"), ("mage", "MAGE")):
        c = cells[("mirror", bench)]
        groups.append((label, [c["auroc_mean_pooled"], c["deployed"]["fnr"]]))

    return grouped_bars(
        groups,
        series_names=["AUROC (can it rank?)", "Miss rate at the deployed threshold"],
        y_max=1.0,
        y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        # A held-in AUROC of 0.999895 must not print as "1". Rounding a near-perfect
        # score to a perfect one is the single most flattering rounding available here.
        fmt=lambda v: f"{v:.4f}" if v >= 0.9995 else f"{v:.3f}",
        title="Matched-mirrors arm, held-in against three unseen corpora",
        colors=[S1, CRIT],
        note="Same model, same threshold (0.998252). Ranking degrades gently. "
             "The operating point collapses.",
    )


def chart_adversarial():
    """Two cheap edits, and what each one costs the detector."""
    data = json.loads((ART / "adversarial_forge_min_mirror.json").read_text())
    labels = {
        "homoglyph_substitute": "Homoglyphs",
        "case_perturb": "Random case",
    }
    groups = [("No attack", [0.0])]
    for r in data["results"]:
        name = labels.get(r["attack"], r["attack"])
        groups.append((f"{name}\n{r['severity']:.0%} of chars", [r["fnr_raw"]]))

    return grouped_bars(
        groups,
        series_names=["Miss rate on 250 AI documents"],
        y_max=1.0,
        y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        fmt=lambda v: f"{v:.1%}" if v else "0%",
        title="What it costs to get AI text past this detector",
        colors=[S2],
        note="Randomising the case of one character in ten is enough. "
             "Unicode normalisation defuses none of it.",
    )


def chart_scaling():
    """The same sweep at two sizes, which is where the speedup column stopped surviving."""
    pilot = json.loads((ART / "spark" / "spark_summary.json").read_text())
    scale = json.loads((ART / "spark_reserve" / "spark_summary.json").read_text())
    by_p = {r["partitions"]: r["speedup"] for r in pilot["rows"]}
    by_s = {r["partitions"]: r["speedup"] for r in scale["rows"]}

    groups = [(f"{p} partition{'s' if p > 1 else ''}", [by_p[p], by_s[p]])
              for p in (1, 2, 4)]
    return grouped_bars(
        groups,
        series_names=["40 documents (pilot)", "2,000 documents (reserve pool)"],
        y_max=1.6,
        y_ticks=[0, 0.5, 1.0, 1.5],
        fmt=lambda v: f"{v:.2f}",
        title="Spark local: the same sweep, scanning 50x more",
        colors=[MUTED, S1],
        note="Ideal here is 1.00, not the partition count: the thread budget is fixed. "
             "The pilot was measuring startup.",
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    write("chart_generalisation.svg", chart_generalisation())
    write("chart_adversarial.svg", chart_adversarial())
    write("chart_scaling.svg", chart_scaling())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
