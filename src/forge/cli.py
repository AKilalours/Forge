"""FORGE command line entrypoint.

Commands that belong to phases which have not started fail loudly with the phase
they need, rather than pretending to run. A stub that silently returns success is
worse than no stub, because it hides that the work has not been done.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from forge.common import config as cfg

app = typer.Typer(add_completion=False, help="FORGE: failure-driven AI-content detection")


class PhaseNotImplemented(RuntimeError):
    pass


def _not_yet(phase: str, what: str) -> None:
    raise PhaseNotImplemented(
        f"{what} is Phase {phase} and is not implemented yet. "
        f"See docs/roadmap.md for what has to land first."
    )


@app.command("spec-check")
def spec_check() -> None:
    """Fail if any config has drifted from the frozen Data Spec v1."""
    problems = cfg.spec_check()
    if problems:
        typer.secho(f"spec-check FAILED ({len(problems)} violation(s)):", fg=typer.colors.RED, bold=True)
        for p in problems:
            typer.echo(f"  - {p}")
        sys.exit(1)
    typer.secho("spec-check OK: configs match data_spec_v1", fg=typer.colors.GREEN)


@app.command()
def ingest(
    config: str = typer.Option(..., "--config"),
    out: str = typer.Option("data/silver", "--out", help="output root for Parquet"),
    total: int = typer.Option(None, "--total", help="override the document target (smoke runs)"),
    with_reserve: bool = typer.Option(False, "--with-reserve", help="also build the mining reserve pool"),
    reserve_out: str = typer.Option("data/reserve", "--reserve-out"),
    read_multiplier: int = typer.Option(20, "--read-multiplier",
                                        help="documents to stream per document kept"),
) -> None:
    """Phase 1: build FORGE-HUMAN from the configured sources."""
    from forge.ingestion.run import ingest as _ingest

    result = _ingest(config, out_root=out, total=total, with_reserve=with_reserve,
                     reserve_root=reserve_out, read_multiplier=read_multiplier)
    typer.secho(f"kept {len(result.docs)} documents", fg=typer.colors.GREEN)
    for sid, st in result.stats_by_source.items():
        typer.echo(f"  {sid}: seen={st['seen']} kept={st['kept']} rejected={st['rejected']}")
    for part, n in sorted(result.partitions.items()):
        typer.echo(f"  parquet {part}: {n}")
    if result.reserve_docs:
        typer.echo(f"  reserve pool: {result.reserve_docs} documents -> {result.reserve_partitions}")
    typer.echo(f"  manifest: {result.manifest_path}")
    for sid, sf in result.shortfalls.items():
        typer.secho(
            f"  SHORTFALL {sid}: wanted {sf['wanted']}, got {sf['got']} "
            f"(keep rate {sf['keep_rate']:.1%})", fg=typer.colors.YELLOW,
        )
        typer.echo(f"     {sf['hint']}")


@app.command("pin-revisions")
def pin_revisions(
    config: str = typer.Option("configs/generation/generators_minimal.yaml", "--config"),
    write: bool = typer.Option(False, "--write", help="rewrite the config in place"),
) -> None:
    """Resolve each generator's HuggingFace repo revision and pin it.

    'Generated with Qwen 3B' is not reproducible: the upstream repo can move and the same
    config would then produce different data under the same dataset version. This
    resolves the current commit sha for every open-source family so the roster is fixed.
    """
    from pathlib import Path

    from huggingface_hub import HfApi

    from forge.generation.pinning import pin_revision, unpinned_lines

    api = HfApi()
    conf = cfg.load(config)
    path = Path(cfg.REPO_ROOT / config)
    text = path.read_text()
    resolved: dict[str, str] = {}
    written: dict[str, str] = {}
    unresolved: list[str] = []
    for fam in conf.get("families", []):
        if fam.get("provider") != "open_source":
            continue
        try:
            sha = api.model_info(fam["model_id"]).sha
        except Exception as e:
            typer.secho(f"  {fam['family']:<10} FAILED  {fam['model_id']}: {e}", fg=typer.colors.RED)
            typer.echo("     gated model? accept its license on huggingface.co and set HF_TOKEN")
            unresolved.append(fam["family"])
            continue
        resolved[fam["family"]] = sha
        # Anchor on the model_id line and rewrite whatever the NEXT revision: value is.
        #
        # This used to be a literal str.replace of "model_id: <id>\n    revision: TODO...",
        # which silently did nothing whenever the model_id line carried a trailing comment
        # (the gated entries do) and could not re-pin an already-pinned roster. str.replace
        # reports a no-op as success, so the command printed a count of API lookups and
        # looked like it had written a file it had not touched.
        text, n = pin_revision(text, fam["model_id"], sha)
        if n != 1:
            typer.secho(
                f"  {fam['family']:<10} RESOLVED BUT NOT WRITTEN: matched {n} revision lines "
                f"for {fam['model_id']} (expected exactly 1)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        written[fam["family"]] = sha
        typer.echo(f"  {fam['family']:<10} {fam['model_id']:<45} {sha}")

    assert written.keys() == resolved.keys()
    if write:
        path.write_text(text)
        # Report substitutions, never lookups. The count has to be derived from the
        # mutation itself or it cannot detect a write that did not happen.
        typer.secho(f"pinned {len(written)} revisions into {config}", fg=typer.colors.GREEN)
        left = unpinned_lines(path.read_text())
        if left:
            typer.secho(
                f"{len(left)} revision(s) still unpinned; unresolved families: "
                f"{unresolved or 'none'}",
                fg=typer.colors.YELLOW,
            )
            if not unresolved:
                typer.secho(
                    "no family failed to resolve, so these lines should have been "
                    "written. This is a bug, not a gating problem.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
    else:
        typer.secho("dry run; pass --write to update the config", fg=typer.colors.YELLOW)


@app.command()
def mirror(
    config: str = typer.Option(..., "--config"),
    humans: str = typer.Option("data/silver", "--humans", help="human parquet root"),
    out: str = typer.Option("data/silver/mirrors", "--out"),
    backend: str = typer.Option("fake", "--backend", help="fake | vllm | transformers"),
    limit: int = typer.Option(None, "--limit", help="cap human documents (smoke runs)"),
    only_family: str = typer.Option(
        None, "--only-family",
        help="mirror just this held-in family's share; the other families' documents are "
             "assigned as usual and left for another invocation",
    ),
) -> None:
    """Phase 2: generate synthetic mirrors."""
    from forge.generation.run import run as _run

    if backend == "fake":
        typer.secho(
            "backend=fake: output is a pipeline test, NOT training data.",
            fg=typer.colors.YELLOW,
        )
    result = _run(config, humans_root=humans, out_root=out, backend=backend, limit=limit,
                  only_family=only_family)
    typer.secho(f"generated {len(result.docs)} mirrors", fg=typer.colors.GREEN)
    for k, v in result.stats.items():
        typer.echo(f"  {k}: {v}")
    for split, n in sorted(result.partitions.items()):
        typer.echo(f"  parquet split={split}: {n}")


@app.command("generate-random")
def generate_random_cmd(
    n: int = typer.Option(..., "--n", help="must EQUAL the mirror count; equal budget per arm"),
    humans: str = typer.Option("data/silver", "--humans"),
    generators: str = typer.Option("configs/generation/generators_minimal.yaml", "--generators"),
    out: str = typer.Option("data/silver/random", "--out"),
    backend: str = typer.Option("fake", "--backend", help="fake | vllm | transformers"),
    only_family: str = typer.Option(
        None, "--only-family",
        help="generate just this held-in family's share of --n; the other families' "
             "documents are assigned as usual and left for another invocation",
    ),
    validation_from: str = typer.Option(
        "configs/generation/mirror_minimal.yaml", "--validation-from",
        help="take the length band and retry count from THIS config, which is arm B's, "
             "so both arms are validated identically and differ only in matching",
    ),
) -> None:
    """Arm A: conventional random synthetic data. The control the project measures against.

    Unmatched AI text from a fixed topic list. Same generators, same decoding grid and the
    same document count as the mirror arm, so any difference between arms is attributable
    to matching rather than to budget.
    """
    from forge.generation.random_synthetic import generate_random, length_pool_from_corpus
    from forge.generation.run import write_mirrors

    if backend == "fake":
        typer.secho("backend=fake: output is a pipeline test, NOT training data.",
                    fg=typer.colors.YELLOW)
    pool = length_pool_from_corpus(humans)
    # ARM B'S VALIDATION BLOCK, ON PURPOSE. Arm A used to fall back to the looser
    # signature defaults (0.5 to 2.0 against arm B's 0.6 to 1.6), which meant the control
    # arm accepted longer documents than the mirror arm and the two arms differed in
    # something other than matching.
    validation = cfg.load(validation_from).get("validation", {})
    res = generate_random(n, pool, cfg.load(generators), backend=backend,
                          only_family=only_family, validation=validation)
    # THE PART NAME CARRIES THE FAMILY so four one-family runs accumulate under one root
    # rather than each overwriting the last. --n stays the FULL corpus size in every one
    # of them: it is what documents are assigned against, not what this call produces.
    parts = write_mirrors(res.docs, out, part=only_family or "000")
    typer.secho(f"generated {len(res.docs)} random-synthetic documents", fg=typer.colors.GREEN)
    for k, v in res.stats.items():
        typer.echo(f"  {k}: {v}")
    for split, n_ in sorted(parts.items()):
        typer.echo(f"  parquet split={split}: {n_}")


@app.command()
def train(
    config: str = typer.Option("configs/minimal/training_mirror.yaml", "--config"),
    smoke: bool = typer.Option(False, "--smoke", help="tiny CPU run that exercises the whole path"),
    resume: str = typer.Option(None, "--resume", help="checkpoint to resume from"),
) -> None:
    """Train a detector. Run --smoke first: it is the cheapest way to find a bug."""
    from forge.training.train import run as _run

    report = _run(config, smoke=smoke, resume=resume)
    f = report.get("final", {})
    if f:
        typer.secho(
            f"FPR@{report['fpr_budget']}={f.get('fpr'):.5f}  FNR={f.get('fnr'):.4f}  "
            f"AUROC={f.get('auroc'):.4f}  ECE={f.get('ece'):.4f}",
            fg=typer.colors.GREEN,
        )
        if not f.get("fpr_measurable", True):
            typer.secho(
                "  NOTE: too few human documents to resolve this FPR budget. "
                "The FPR figure is not evidence.",
                fg=typer.colors.YELLOW,
            )


@app.command()
def evaluate(
    config: str = typer.Option(..., "--config"),
    arm: str = typer.Option(None, "--arm", help="which trained arm to attack; default: the config's"),
    limit: int = typer.Option(500, "--limit", help="AI test documents to attack"),
    out: str = typer.Option("reports/experiments", "--out"),
    attacks: str = typer.Option(None, "--attacks", help="comma-separated subset"),
    human_limit: int = typer.Option(
        0, "--human-limit",
        help="human documents to score per condition, for the false positive column. "
             "0 means use --limit. FPR RESOLUTION IS 1/n: 500 documents cannot tell 0.001 "
             "from 0.000, which is the budget the threshold was fitted at, but they "
             "resolve a catastrophic rise easily, and that is what a score-inflating "
             "transform produces."),
    conditions: str = typer.Option(
        None, "--conditions",
        help="comma-separated preprocessing conditions to score. Default 'raw,normalised'. "
             "Add 'folded' to measure confusables folding against homoglyph substitution, "
             "and 'casefolded' to measure case folding against case perturbation. Each "
             "extra condition rescores every cell AND the clean baseline, so it roughly "
             "doubles the run; the clean column is what says whether the defence is free."),
) -> None:
    """Run the adversarial laboratory against a trained arm: the delta-FNR table.

    Attacks are applied to AI documents from the arm's own TEST split, because FNR is the
    metric an evader moves and the test split is the only data the model has not seen.

    Each attack is measured twice, raw and after the production normalisation pass, so the
    table separates "this attack beats the model" from "this attack beats a deployment that
    skipped normalisation". No-ops and attacks that fail the meaning-preservation check are
    excluded from the scores and counted separately, because an attack that mangles the text
    into nonsense trivially defeats a detector and proves nothing.

    This was `_not_yet("3", "the evaluation lab")` while attacks.py and lab.py were both
    written and tested. The held-out-generator evaluation is a separate entrypoint,
    scripts/eval_ood.py; this command does not duplicate it.
    """
    import json

    from forge.adversarial.lab import (
        DEFAULT_CONDITIONS,
        render_cost,
        render_table,
        run_attacks,
    )
    from forge.inference.scorer import ArmUnavailable, load_arm
    from forge.training.data import load_examples

    cfg_data = cfg.load(config)
    arm_name = arm or ("mirror" if cfg_data["data"].get("arm") != "random" else "baseline")
    try:
        loaded = load_arm(arm_name)
    except ArmUnavailable as error:
        raise PhaseNotImplemented(
            f"the adversarial lab attacks a trained detector: {error}"
        ) from None

    # HUMAN ROOT IS NOT OPTIONAL HERE, and leaving it out is what made the first run of
    # the four-condition sweep unreadable. Without it load_examples returns AI documents
    # only, the human cost of every condition is empty, and a transform that inflates all
    # scores reports a flawless FNR with nothing to contradict it.
    examples = load_examples(
        human_root=cfg_data["paths"]["human"],
        ai_root=cfg_data["paths"]["ai"], splits=("test",),
        expect_arm=cfg_data["data"].get("arm"),
    )
    ai = [e for e in examples if e.label == 1][:limit]
    # HUMAN DOCUMENTS ARE NOT OPTIONAL CONTEXT. FNR alone cannot separate a defence from a
    # transform that inflates every score, and the cost of inflation lands here as false
    # positives against a threshold fitted on untransformed text.
    human = [e for e in examples if e.label == 0][:human_limit or limit]
    if not ai:
        raise PhaseNotImplemented(
            f"no AI documents in the test split of {cfg_data['paths']['ai']}. The lab "
            "measures how far an evader can push FNR, and FNR needs AI documents."
        )

    typer.echo(
        f"attacking {len(ai)} AI test documents with {loaded.policy.model_version} "
        f"at threshold {loaded.policy.threshold:.6f}"
    )
    if not human:
        typer.echo(
            "WARNING: no human documents in this split. Every number below is FNR on AI "
            "documents, and a condition that inflates all scores will read as a perfect "
            "defence. Treat any new condition as unmeasured."
        )
    results, cost = run_attacks(
        [e.text for e in ai], [e.doc_id for e in ai],
        score_fn=lambda texts: [loaded.score(t).mean for t in texts],
        threshold=loaded.policy.threshold,
        attacks=[a.strip() for a in attacks.split(",")] if attacks else None,
        conditions=(tuple(c.strip() for c in conditions.split(","))
                    if conditions else DEFAULT_CONDITIONS),
        human_texts=[e.text for e in human],
    )
    typer.echo("\n" + render_table(results))
    typer.echo("\n" + render_cost(cost))

    path = Path(out) / f"adversarial_{loaded.experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment": loaded.experiment,
        "model_version": loaded.policy.model_version,
        "threshold": loaded.policy.threshold,
        "fpr_budget": loaded.policy.fpr_budget,
        "n_ai_documents": len(ai),
        "split": "test",
        "conditions_scored": list(results[0].fnr) if results else [],
        "human_cost": {c: {"fpr": round(v.fpr, 6), "n_human": v.n_human,
                           "median": round(v.median, 6), "p95": round(v.p95, 6),
                           "p99": round(v.p99, 6), "max": round(v.maximum, 6)}
                       for c, v in cost.items()},
        "note": (
            "delta-FNR against the clean baseline MEASURED UNDER THE SAME CONDITION, not "
            "against one global baseline. No-ops and attacks failing preserves_meaning are "
            "excluded from the scores and counted separately. Every condition carries its "
            "own clean_fnr because folding and casefolding move clean documents too, and a "
            "defence is only worth adopting if it lowers the attacked FNR by more than it "
            "raises the clean one. Every rate here is measured on AI documents only; the "
            "human cost of each condition is under human_cost, and a condition that lowers "
            "the miss rate by moving every score has not defended anything."
        ),
        "results": [r.as_dict() for r in results],
    }, indent=2) + "\n")
    typer.echo(f"\nwritten {path}")


@app.command()
def mine(
    config: str = typer.Option(..., "--config"),
    reserve: str = typer.Option("data/reserve", "--reserve", help="reserve pool parquet root"),
    out: str = typer.Option("reports/experiments", "--out"),
    arm: str = typer.Option("mirror", "--arm", help="which trained arm mines: baseline|mirror"),
    round_name: str = typer.Option("mining_run_001", "--round"),
    limit: int = typer.Option(0, "--limit", help="cap reserve documents scanned, 0 for all"),
    min_confidence: float = typer.Option(
        None, "--min-confidence",
        help="mining gate. Defaults to the arm's own deployed threshold, which is the "
             "loosest value that is still valid: a gate below it selects documents "
             "production never called AI.",
    ),
) -> None:
    """Phase 4: hard negative mining pass over the human reserve pool.

    Scans reserve documents with a trained arm, keeps the ones it calls AI at high
    confidence, clusters those failures into an atlas and selects across clusters. Writes
    the round to `--out`.

    This used to load the config, raise if --model was absent, and RETURN if it was present,
    so a caller who satisfied the precondition got exit code 0 and no work. The round was
    never invoked. `run_round` is called below.

    Mining scores with the same aggregation evaluation uses, mean over windows, because the
    false positives being characterised have to be the ones production would make.
    """
    from forge.hard_negative.reserve import CheckpointScorer, load_reserve
    from forge.hard_negative.run import run_round
    from forge.inference.scorer import ArmUnavailable

    cfg.load(config)
    try:
        scorer = CheckpointScorer(arm)
    except ArmUnavailable as error:
        raise PhaseNotImplemented(
            f"mining needs a trained detector to score the reserve pool: {error}"
        ) from None

    # THE GATE DEFAULTS TO THE ARM'S THRESHOLD, and is checked BEFORE the reserve pool is
    # read. This used to default to 0.90, a number that predates the calibrated
    # thresholds: every arm now deploys above 0.996, so the default was guaranteed to
    # raise. It raised inside scan(), after loading a 735 MB checkpoint and every document
    # in the pool, which is a minute of work to be told the arguments were wrong. The
    # docstring on that check is right and stays; what was wrong was arriving there.
    if min_confidence is None:
        min_confidence = scorer.threshold
    if min_confidence < scorer.threshold:
        raise typer.BadParameter(
            f"--min-confidence {min_confidence} is below {arm}'s deployed threshold "
            f"{scorer.threshold:.6f}. The mining gate must be at least as strict as the "
            f"production decision, or it selects documents production never called AI. "
            f"Omit the flag to use the threshold itself.",
            param_hint="--min-confidence",
        )

    docs = load_reserve(reserve, limit or None)
    typer.echo(
        f"scanning {len(docs)} reserve documents with {scorer.model_version} "
        f"at operating threshold {scorer.threshold:.6f}, mining gate {min_confidence:.6f}"
    )
    round_ = run_round(
        docs, scorer,
        operating_threshold=scorer.threshold,
        round_name=round_name,
        min_confidence=min_confidence,
    )
    path = round_.write(Path(out) / f"{round_name}.json")
    stats = round_.stats.as_dict()
    typer.echo(
        f"scanned {stats['scanned']}, "
        f"errors at operating threshold {stats['errors_at_operating_threshold']} "
        f"({stats['error_rate']:.4%}), "
        f"above the confidence gate {stats['above_confidence_gate']}, "
        f"selected {stats['selected']}"
    )
    typer.echo(f"train refs {len(round_.train_refs)}, holdout refs {len(round_.holdout_refs)}")
    typer.echo(f"written {path}")


@app.command()
def gate(
    model_version: str = typer.Option(..., "--model-version"),
    eval_report: str = typer.Option(None, "--eval-report", help="JSON from the evaluation lab"),
    config: str = typer.Option("configs/eval/regimes.yaml", "--config"),
) -> None:
    """Run the release gate against a candidate model's evaluation report."""
    import json
    from pathlib import Path

    from forge.evaluation.release_gate import evaluate

    gate_cfg = cfg.load(config)["release_gate"]
    if eval_report is None:
        raise PhaseNotImplemented(
            "the release gate needs an evaluation report to judge. The gate policy is "
            "implemented and tested; produce a report with `forge evaluate` once a model "
            "exists, then pass it with --eval-report."
        )
    report = json.loads(Path(eval_report).read_text())
    metrics = report.get("headline") or report
    result = evaluate(metrics, gate_cfg)
    if result.passed:
        typer.secho(f"GATE PASSED: {model_version} may be promoted to canary", fg=typer.colors.GREEN)
        return
    typer.secho(f"GATE FAILED: {model_version} must not ship", fg=typer.colors.RED, bold=True)
    for f in result.failures:
        typer.echo(f"  - {f}")
    sys.exit(1)


if __name__ == "__main__":
    app()
