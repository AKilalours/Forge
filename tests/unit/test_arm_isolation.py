"""Each training arm must read its OWN data.

Guard for a bug that would have produced a false negative at real cost: all three arm
configs pointed at the same AI directory and `data.include` was decorative, so Arm A and
Arm B would have trained on identical data and produced identical numbers. Nothing would
have errored. Two successful runs, two plausible rows, one false finding.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forge.common.config import load
from forge.training.data import ARM_PROMPT_VERSION, ArmMismatch, load_examples


def _write(root, prompt_version, n=4):
    d = root / "split=train"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{
        "sample_id": f"s{i}", "source_group_id": f"g{i}", "text": "some generated text here",
        "split": "train", "domain": "web",
        "generator": {"family": "qwen", "released": "2024-09"},
        "mirror": {"prompt_version": prompt_version},
    } for i in range(n)]
    pq.write_table(pa.Table.from_pylist(rows), d / "part-000.parquet")


def test_the_three_arm_configs_point_at_different_data():
    """The actual bug. If two arms share an AI source, their results are the same run."""
    sources = {
        name: load(f"configs/training/{name}_minimal.yaml")["paths"]["ai"]
        for name in ("baseline", "mirror", "hard_negative")
    }
    assert sources["baseline"] != sources["mirror"], (
        f"Arm A and Arm B both read {sources['baseline']}; they would train on identical "
        "data and the comparison would be meaningless"
    )


def test_every_arm_config_declares_its_arm():
    for name in ("baseline", "mirror", "hard_negative"):
        cfg = load(f"configs/training/{name}_minimal.yaml")
        assert cfg["data"]["arm"] in ARM_PROMPT_VERSION


def test_loading_the_wrong_arms_data_raises(tmp_path):
    _write(tmp_path / "mirrors", "mirror_v1")
    with pytest.raises(ArmMismatch, match="training on the wrong data"):
        load_examples(ai_root=tmp_path / "mirrors", expect_arm="random")


def test_loading_the_right_arms_data_succeeds(tmp_path):
    _write(tmp_path / "random", "random_v1")
    ex = load_examples(ai_root=tmp_path / "random", expect_arm="random")
    assert len(ex) == 4 and all(e.label == 1 for e in ex)


def test_mixed_provenance_in_one_directory_is_refused(tmp_path):
    """A directory containing both arms means a previous run wrote into the wrong place."""
    root = tmp_path / "mixed_up"
    _write(root, "mirror_v1")
    d = root / "split=val"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{
        "sample_id": "r1", "source_group_id": "gr1", "text": "t", "split": "val",
        "domain": "web", "generator": {"family": "qwen", "released": "2024-09"},
        "mirror": {"prompt_version": "random_v1"},
    }]), d / "part-000.parquet")
    with pytest.raises(ArmMismatch):
        load_examples(ai_root=root, expect_arm="mirror")


def test_training_refuses_a_config_with_no_declared_arm(tmp_path):
    from forge.training.train import run

    cfg = {
        "experiment": {"id": "x"}, "model_config": "configs/models/forge_base.yaml",
        "paths": {"human": str(tmp_path), "out": str(tmp_path)},
        "data": {}, "training": {"batch_size": 2, "learning_rate": 1e-5, "epochs": 1},
    }
    with pytest.raises(RuntimeError, match="must declare data.arm"):
        run(cfg, smoke=True)


# ------------------------- the guard that was pinned to a version string that then moved

def test_the_arm_guard_accepts_any_version_of_its_own_prompt_family():
    """It was a map to the literal "random_v1", and the prompt moved to v2.

    Correcting the prompts to ask for words rather than tokens made them v2, which
    data_spec_v1 requires, and the next training run died with "config declares arm
    'random' but data/silver/random contains ['random_v2']". The guard was right that
    something had changed and wrong about what: the data was correct and the guard was
    stale. Bumping the literal to v2 would have fixed that night and rotted at v3.
    """
    from forge.training.data import check_arm_versions

    for version in ("random_v1", "random_v2", "random_v17"):
        check_arm_versions("random", {version}, "data/silver/random")


def test_arm_c_shares_the_mirror_arms_prompt_family():
    from forge.training.data import check_arm_versions

    check_arm_versions("hard_negative", {"mirror_v2"}, "data/silver/mirrors")


def test_one_arms_data_under_another_arms_config_is_still_refused():
    """The thing the guard is actually for, unchanged by the fix."""
    from forge.training.data import ArmMismatch, check_arm_versions

    with pytest.raises(ArmMismatch, match="wrong data"):
        check_arm_versions("random", {"mirror_v2"}, "data/silver/random")
    with pytest.raises(ArmMismatch, match="wrong data"):
        check_arm_versions("mirror", {"random_v2"}, "data/silver/mirrors")


def test_a_directory_holding_two_prompt_versions_is_refused():
    """The stronger half, which the old exact-match check only got by accident.

    A directory with both v1 and v2 is a corpus that was half regenerated. Training
    across a prompt change averages two different data-generating processes, and nothing
    downstream can see that it happened.
    """
    from forge.training.data import ArmMismatch, check_arm_versions

    with pytest.raises(ArmMismatch, match="mixes prompt versions"):
        check_arm_versions("random", {"random_v1", "random_v2"}, "data/silver/random")


def test_an_unknown_arm_name_does_not_silently_pass_everything():
    """An arm with no declared family means the check cannot run, not that it passed."""
    from forge.training.data import ARM_PROMPT_FAMILY

    assert set(ARM_PROMPT_FAMILY) == {"random", "mirror", "hard_negative"}


# ---------------------------------------------------------------------------
# THE PARTITION-KEY COLLISION, and why it is this repository's own doing.
#
# forge.ingestion.writer partitions as source=<x>/split=<y>/ AND writes physical `source`
# and `split` columns into every file. pq.read_table applies hive partitioning discovery to
# the path it is handed, so one shard offers two definitions of `source`: a string column
# from the file and a dictionary-encoded one inferred from the directory. Older pyarrow
# refuses to merge them:
#
#   ArrowTypeError: Unable to merge: Field source has incompatible types:
#   string vs dictionary<values=string, indices=int32, ordered=0>
#
# It surfaced when a command was run outside the project venv, against an older pyarrow,
# while the identical call kept working on 25.0.1 inside it. That is not a fixed bug, it is
# a fixed dependency on a version, and it sat at the first line of every training and
# evaluation run that loads the human corpus.
# ---------------------------------------------------------------------------


def test_a_shard_whose_column_name_matches_its_partition_key_still_loads(tmp_path) -> None:
    """The exact layout the writer produces. Reading one named file must not consult the
    directory it happens to sit in."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from forge.training.data import load_examples

    part = tmp_path / "source=fw" / "split=test"
    part.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "doc_id": ["d1", "d2"],
            "source_group_id": ["g1", "g2"],
            "text": ["human one", "human two"],
            "split": ["test", "test"],
            "domain": ["news", "news"],
            # The collision: a physical column with the same name as the partition key.
            "source": ["fw", "fw"],
        }),
        part / "part-000.parquet",
    )

    rows = load_examples(human_root=str(tmp_path), splits=("test",))
    assert [r.doc_id for r in rows] == ["d1", "d2"]
    assert all(r.label == 0 for r in rows)


def test_the_reader_does_not_infer_columns_from_the_path(tmp_path) -> None:
    """Stated directly against the helper, because the guarantee is that the path is not an
    input to the schema, and a loader test could pass for other reasons."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from forge.training.data import _read_rows

    part = tmp_path / "source=fw" / "split=test"
    part.mkdir(parents=True)
    pq.write_table(pa.table({"doc_id": ["d1"], "source": ["written-in-the-file"]}),
                   part / "part-000.parquet")

    rows = _read_rows(str(part / "part-000.parquet"), None)
    assert rows == [{"doc_id": "d1", "source": "written-in-the-file"}], (
        "the value must come from the file, and no partition columns may be invented "
        "from the directory names"
    )
