"""The flywheel DAG must load, be shaped correctly, and not be expensive to parse.

WHY THESE TESTS AND NOT OTHERS. A DAG file is not ordinary code: Airflow's scheduler
re-imports it on a loop, and a DAG that imports cleanly on a laptop can still be wrong in
ways that cost real money in a deployment. The three that matter here:

1. IT MUST IMPORT. A DagBag import error means the DAG silently does not exist. Airflow
   shows it as a broken-DAG banner in a UI nobody is watching at 3am, and the scheduled run
   simply never happens. This repo has already found four mechanisms that were wired up and
   never ran; a DAG that fails to parse is the same failure with a scheduler attached.

2. IT MUST PARSE CHEAPLY. The scheduler imports every DAG file every few seconds. `import
   torch` at module scope turns that into seconds of CPU and a gigabyte of RSS in the
   scheduling process. That is why every step is a BashOperator shelling out to the CLI
   rather than a PythonOperator importing forge.

3. THE EXPENSIVE DEFAULTS MUST BE OVERRIDDEN. catchup defaults to True, which backfills
   every missed interval on first deploy. For a pipeline whose middle step is a training
   job, deploying a @weekly DAG with a start_date three months back queues twelve
   retrainings. Nobody notices until the bill arrives.

These run without a scheduler, a database or an executor. DagBag parses the file the same
way Airflow does, which is the part worth checking.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DAG_DIR = ROOT / "orchestration" / "dags"
DAG_ID = "forge_flywheel"

pytest.importorskip("airflow", reason="apache-airflow is in the [orchestration] extra")


@pytest.fixture(scope="module")
def dagbag():
    """Load the DAG folder the way Airflow does, across the 2.x / 3.x API break.

    Airflow 3 moved DagBag from airflow.models.dagbag to airflow.dag_processing.dagbag and
    dropped the include_examples keyword. Pinning to either spelling makes this file pass
    on one major version and error on the other, which for a test whose whole job is "does
    this DAG load" would be the joke writing itself.
    """
    import inspect

    try:
        from airflow.dag_processing.dagbag import DagBag  # Airflow 3
    except ImportError:                                            # pragma: no cover
        from airflow.models.dagbag import DagBag  # Airflow 2

    kwargs = {"dag_folder": str(DAG_DIR)}
    if "include_examples" in inspect.signature(DagBag.__init__).parameters:
        kwargs["include_examples"] = False
    return DagBag(**kwargs)


def test_the_dag_file_imports_without_errors(dagbag) -> None:
    """A broken DAG is invisible: Airflow shows a banner and the run never happens."""
    assert not dagbag.import_errors, (
        "the DAG failed to import, so Airflow would show it as broken and never run it:\n"
        + "\n".join(f"{k}: {v}" for k, v in dagbag.import_errors.items())
    )
    assert DAG_ID in dagbag.dags, f"expected {DAG_ID}, found {list(dagbag.dags)}"


def test_parsing_is_cheap_enough_for_a_scheduler_loop() -> None:
    """THE REGRESSION THIS PREVENTS: someone converts a step to PythonOperator and imports
    forge, or torch, at module scope. It works, tests pass, and the scheduler starts
    spending seconds per parse cycle on a DAG it parses constantly."""
    import subprocess
    import sys

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import runpy; runpy.run_path({str(DAG_DIR / 'forge_flywheel.py')!r})"],
        capture_output=True, text=True, timeout=120,
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert elapsed < 10.0, (
        f"the DAG file took {elapsed:.1f}s to import. The scheduler does this every few "
        "seconds. Something heavy is being imported at module scope; move it inside a task."
    )


HEAVY_MODULES = {"torch", "transformers", "forge", "datasets", "sklearn", "scipy"}


def _top_level_imports(tree) -> set[str]:
    """Modules imported at MODULE scope. An import inside a function is fine: the
    scheduler pays for it only when that function runs, which is never during parsing."""
    import ast

    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    return set(found)


def test_the_dag_does_not_import_heavy_modules_at_module_scope() -> None:
    """Stated separately from the timing check, because the timer is a proxy and this is
    the rule. A fast machine could pass the timer and still be wrong.

    THIS TEST FAILED ON ITS OWN DOCUMENTATION. The first version searched the file's TEXT
    for "import torch", and the DAG's module docstring contains that phrase in the sentence
    explaining why you must not do it. Filtering out "#" lines did not help: the offending
    text was inside a docstring.

    The fix is not smarter string filtering, which would break again on the next comment
    that quotes code. It is to stop reading the file as text and parse it. ast.walk sees
    import STATEMENTS and cannot be confused by prose that mentions one. Any check that
    greps source code is a check a comment can break.
    """
    import ast

    tree = ast.parse((DAG_DIR / "forge_flywheel.py").read_text())
    heavy = HEAVY_MODULES & _top_level_imports(tree)
    assert not heavy, (
        f"{sorted(heavy)} imported at DAG scope. The scheduler re-imports this file on a "
        "loop; heavy imports belong inside a task, or in the CLI the task shells out to."
    )


def test_the_import_check_is_not_fooled_by_a_comment() -> None:
    """The regression for the check itself.

    A module that only MENTIONS torch in a docstring must pass; one that imports it must
    fail. Without this, the next person who "fixes" the check by going back to string
    matching has nothing telling them why that was wrong the first time.
    """
    import ast

    mentions_only = ast.parse('"""Never write import torch at module scope."""\nimport os\n')
    actually_imports = ast.parse("import torch\n")
    inside_a_function = ast.parse("def task():\n    import torch\n    return torch\n")

    assert not HEAVY_MODULES & _top_level_imports(mentions_only)
    assert HEAVY_MODULES & _top_level_imports(actually_imports)
    assert not HEAVY_MODULES & _top_level_imports(inside_a_function), (
        "an import inside a task function costs nothing at parse time and must be allowed"
    )


def test_catchup_is_off(dagbag) -> None:
    """THE MOST EXPENSIVE DEFAULT IN AIRFLOW.

    catchup defaults to True. A @weekly DAG deployed with a start_date three months back
    immediately queues twelve backfill runs, each of which retrains a model.
    """
    dag = dagbag.dags[DAG_ID]
    assert dag.catchup is False, (
        "catchup is on, so first deploy will backfill every interval since start_date and "
        "retrain once per interval"
    )


def test_only_one_round_runs_at_a_time(dagbag) -> None:
    """Two concurrent rounds mine the same reserve pool and overwrite each other. The
    mined-id ledger exists to stop round two re-finding round one's failures, and that
    guarantee means nothing if the rounds overlap."""
    assert dagbag.dags[DAG_ID].max_active_runs == 1


def test_the_steps_run_in_the_order_the_flywheel_requires(dagbag) -> None:
    """Mine before mirror before train before evaluate before gate. Any other order is a
    different experiment."""
    dag = dagbag.dags[DAG_ID]
    expected = [
        ("start", "mine_reserve_pool"),
        ("mine_reserve_pool", "generate_targeted_mirrors"),
        ("generate_targeted_mirrors", "retrain_candidate"),
        ("retrain_candidate", "evaluate_candidate"),
        ("evaluate_candidate", "release_gate"),
        ("release_gate", "round_complete"),
    ]
    for upstream, downstream in expected:
        assert downstream in dag.get_task(upstream).downstream_task_ids, (
            f"{downstream} does not follow {upstream}"
        )


def test_the_expensive_steps_are_not_retried(dagbag) -> None:
    """A training job that failed on resources fails the same way again, at twice the cost.
    A release gate that failed on the numbers will fail on the same numbers."""
    dag = dagbag.dags[DAG_ID]
    for task_id in ("retrain_candidate", "release_gate"):
        assert dag.get_task(task_id).retries == 0, (
            f"{task_id} is retried; it should fail loudly instead"
        )


def test_the_gate_is_the_last_gate(dagbag) -> None:
    """Nothing may promote a candidate downstream of the gate.

    A flywheel that ships whatever it produced is a conveyor belt. The gate exits non-zero
    when a candidate must not ship, and the only task after it is a marker.
    """
    dag = dagbag.dags[DAG_ID]
    after_gate = dag.get_task("release_gate").downstream_task_ids
    assert after_gate == {"round_complete"}
    from airflow.providers.standard.operators.empty import EmptyOperator

    assert isinstance(dag.get_task("round_complete"), EmptyOperator), (
        "the task after the gate does real work; promotion must not be an automatic "
        "consequence of the gate passing"
    )


def test_every_output_path_is_scoped_to_the_run(dagbag) -> None:
    """Earlier today a flat filename let one machine's records overwrite another's, and it
    was visible only because it produced an impossible number. A scheduler makes that
    routine, so every artifact path carries the run id."""
    dag = dagbag.dags[DAG_ID]
    for task_id in ("mine_reserve_pool", "generate_targeted_mirrors", "retrain_candidate",
                    "evaluate_candidate", "release_gate"):
        command = dag.get_task(task_id).bash_command
        assert "run_id" in command, f"{task_id} writes to a path that two runs could share"


def test_the_dag_imports_without_deprecation_warnings() -> None:
    """Keeps this DAG off an API that is on its way out.

    Airflow 3 still accepts the 2.x spellings and warns. A warning is the worst signal a
    library can send: it runs, so nobody acts on it, right up until a major release removes
    the path and the DAG stops parsing. For a DAG that means the scheduled run silently
    never happens, which is the failure this repository has now found seventeen times.

    Same shape as the unbounded ruff>=0.4 that hid 109 errors: tooling drift is invisible
    while it is merely warning.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-W", "error::UserWarning",
         "-c", f"import runpy; runpy.run_path({str(DAG_DIR / 'forge_flywheel.py')!r})"],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        "the DAG imports something deprecated; move to the current path before a major "
        f"release removes it:\n{proc.stderr[-3000:]}"
    )
