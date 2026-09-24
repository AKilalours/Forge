"""The FORGE flywheel, as an Airflow DAG.

    scan the reserve pool -> cluster the failures -> generate targeted mirrors
      -> retrain -> evaluate -> gate

That loop is the claim this whole project tests: that choosing WHICH synthetic data to
generate, based on where the current detector actually fails, buys more robustness per
example than generating more at random. It is also the only genuinely recurring job here,
which is why it is the one thing in this repo that earns an orchestrator.

WHY BashOperator AND NOT PythonOperator. The scheduler re-parses every DAG file on a short
loop, so anything imported at module scope is imported over and over in the scheduler
process. `import torch` at DAG level would add seconds to every parse and hold a GB of RSS
in a process that is meant to be scheduling, not computing. Shelling out to the `forge` CLI
keeps this file's imports to the standard library and Airflow itself, and it orchestrates
exactly the commands a human runs by hand, which is the point of an orchestrator.

AIRFLOW 3 IMPORT PATHS, ON PURPOSE. DAG comes from airflow.sdk and the operators from
apache-airflow-providers-standard. The 2.x spellings still work in 3.x and emit deprecation
warnings, which is the worst signal a library can send: it runs, so nobody acts on it,
until a major release removes the path and the DAG stops parsing. For a DAG that means the
scheduled run silently never happens.

THREE SETTINGS THAT ARE NOT DEFAULTS, EACH FOR A REASON.

catchup=False. Airflow's default is True: a DAG with a weekly schedule, deployed with a
start_date three months back, immediately queues twelve backfill runs. For a pipeline whose
middle step is a training job, that is twelve retrainings nobody asked for. This is the
single most expensive default in Airflow and it is on by default.

max_active_runs=1. The flywheel is sequential by nature: two concurrent rounds would mine
the same reserve pool, select overlapping documents, and write over each other's outputs.
The ledger in hard_negative/mining.py exists to stop round two re-mining round one's
findings, and that guarantee is worth nothing if the two rounds run at once.

retries=0 on train. Retrying a network blip is sensible. Retrying a three-hour training job
that OOMed will OOM again and bill twice for it. The expensive steps fail loudly instead.

RUN-SCOPED OUTPUT PATHS. Every artifact this DAG writes goes under a directory named for
the run. Earlier today a flat filename let one machine's scaling records overwrite
another's, and the corruption was visible only because it produced an impossible number. A
scheduled DAG makes that failure mode routine rather than occasional, so the paths carry
the run id from the start.
"""

from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG

# Jinja, rendered by Airflow at task run time. Keeping the run id in the path is what makes
# two runs unable to collide.
RUN_DIR = "reports/experiments/flywheel/{{ run_id | replace(':', '-') | replace('+', '-') }}"
ARM = "mirror"

DEFAULT_ARGS = {
    "owner": "forge",
    "depends_on_past": False,
    # A step that fails is a step someone must look at. Mailing on retry as well would
    # train the owner to ignore the mail.
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

with DAG(
    dag_id="forge_flywheel",
    description="Mine the reserve pool for confident false positives, mirror them, retrain, gate.",
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["forge", "phase7", "flywheel"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    mine = BashOperator(
        task_id="mine_reserve_pool",
        bash_command=(
            "forge mine --config configs/training/mirror.yaml "
            f"--arm {ARM} --reserve data/reserve "
            f"--out {RUN_DIR} "
            "--round {{ run_id | replace(':', '-') }}"
            # No --min-confidence. It used to say 0.90, which predates the calibrated
            # thresholds and would have failed every scheduled run on arrival: the arms
            # deploy above 0.996 and the gate may not be looser than the deployment.
            # Omitted, the CLI uses the arm's own threshold, so the DAG cannot drift out
            # of date with a retrained arm the way a literal does.
        ),
        doc_md=(
            "Scores the reserve pool with the deployed arm and keeps documents it calls AI "
            "at high confidence. The reserve pool is human text and disjoint from training "
            "by construction, so every positive here is a false positive, and the "
            "confidence gate is what separates an informative failure from a coin flip "
            "near the threshold."
        ),
    )

    mirror = BashOperator(
        task_id="generate_targeted_mirrors",
        bash_command=(
            "forge mirror --config configs/training/mirror.yaml "
            f"--source {RUN_DIR}/{{{{ run_id | replace(':', '-') }}}}.json "
            f"--out {RUN_DIR}/mirrors"
        ),
        doc_md="Generates mirrors of the mined failures. This is the experiment's one variable.",
    )

    train = BashOperator(
        task_id="retrain_candidate",
        bash_command=(
            "forge train --config configs/training/mirror.yaml "
            f"--extra-ai {RUN_DIR}/mirrors "
            f"--out {RUN_DIR}/candidate"
        ),
        # NOT retried. See the module docstring: a training job that failed on resources
        # fails the same way the second time, at twice the cost.
        retries=0,
        doc_md="Retrains on the original corpus plus the targeted mirrors.",
    )

    evaluate = BashOperator(
        task_id="evaluate_candidate",
        bash_command=(
            "forge evaluate --config configs/eval/regimes.yaml "
            f"--model {RUN_DIR}/candidate "
            f"--out {RUN_DIR}/evaluation.json"
        ),
        doc_md=(
            "In-distribution plus HC3, MAGE and RAID, at the threshold the candidate would "
            "actually deploy at."
        ),
    )

    gate = BashOperator(
        task_id="release_gate",
        bash_command=(
            "forge gate --model-version {{ run_id | replace(':', '-') }} "
            f"--eval-report {RUN_DIR}/evaluation.json "
            "--config configs/eval/regimes.yaml"
        ),
        # The gate is the point of the pipeline. It exits non-zero when the candidate must
        # not ship, and that non-zero must fail the task rather than be swallowed: a
        # flywheel that promotes whatever it produced is not a flywheel, it is a conveyor.
        retries=0,
        doc_md=(
            "Fails the run when the candidate does not clear the release policy. Nothing "
            "downstream of this promotes anything; promotion is a human decision made on a "
            "passed gate, not an automatic consequence of one."
        ),
    )

    done = EmptyOperator(task_id="round_complete")

    start >> mine >> mirror >> train >> evaluate >> gate >> done
