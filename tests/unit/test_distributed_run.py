"""The step semantics of the two strategies, which differ silently when confused.

WHY THIS FILE EXISTS. FSDP and DeepSpeed split a gradient-accumulation step differently:

  FSDP      the CALLER scales the loss by 1/accum, backwards every micro-batch, and steps
            the optimizer once at the boundary.
  DeepSpeed the ENGINE scales by gradient_accumulation_steps and decides when to step; the
            caller calls backward() and step() on every micro-batch.

Cross the two and nothing raises. Use the FSDP loop on a DeepSpeed engine and the loss is
scaled twice, so the effective learning rate is accum times too small. Use the DeepSpeed
loop on FSDP and the optimizer steps accum times per batch, so the real global batch is
per_device_batch and the whole scaling comparison is measuring something else. Both paths
produce a believable throughput number, which for a benchmark is the worst failure mode
available: it does not look broken.

forge.training.scaling holds the global batch fixed ACROSS WORLD SIZES. DistributedRun
holds the step semantics fixed ACROSS STRATEGIES. These tests pin the second one with
fakes, so they run on any machine, with no GPU and without deepspeed installed. That is
deliberate: the bug they describe is arithmetic, and arithmetic does not need an A100.
"""

from __future__ import annotations

import pytest

from forge.training.distributed import DistConfig, DistributedRun


class FakeLoss:
    """Records the divisor it was scaled by before backward()."""

    def __init__(self, sink: list[float], value: float = 1.0) -> None:
        self.sink = sink
        self.value = value

    def __truediv__(self, divisor):
        return FakeLoss(self.sink, self.value / divisor)

    def backward(self) -> None:
        self.sink.append(self.value)


class FakeOptimizer:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroed = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.zeroed += 1


class FakeEngine:
    """Mimics the DeepSpeed engine contract this code depends on, and nothing else."""

    def __init__(self, accum: int) -> None:
        self.accum = accum
        self.backwards: list[float] = []
        self.steps = 0
        self._seen = 0

    def backward(self, loss) -> None:
        # The real engine scales internally. Recording the value it was HANDED is the
        # point: if the caller pre-scales, this list shows fractions.
        self.backwards.append(loss.value)
        self._seen += 1

    def step(self) -> None:
        self.steps += 1

    def is_gradient_accumulation_boundary(self) -> bool:
        return self._seen % self.accum == 0


def test_fsdp_scales_the_loss_by_grad_accum_exactly_once() -> None:
    sink: list[float] = []
    run = DistributedRun(model=None, strategy="fsdp", grad_accum=4, optimizer=FakeOptimizer())
    for _ in range(4):
        run.micro_step(FakeLoss(sink))
    assert sink == [0.25, 0.25, 0.25, 0.25], (
        "each micro-batch must contribute 1/accum of the gradient; anything else changes "
        "the effective learning rate"
    )


def test_fsdp_steps_the_optimizer_once_per_accumulation_boundary() -> None:
    """The regression that would silently shrink the global batch to per_device_batch."""
    opt = FakeOptimizer()
    run = DistributedRun(model=None, strategy="fsdp", grad_accum=4, optimizer=opt)
    stepped = [run.micro_step(FakeLoss([])) for _ in range(8)]
    assert stepped == [False, False, False, True, False, False, False, True]
    assert opt.steps == 2, f"two boundaries in eight micro-batches at accum=4, got {opt.steps}"
    assert opt.zeroed == 2, "gradients must be cleared at the boundary, not left to accumulate"


def test_deepspeed_is_handed_the_raw_loss_and_never_a_prescaled_one() -> None:
    """The regression in the other direction: double scaling.

    The engine divides by gradient_accumulation_steps itself. If this code also divides,
    every gradient is accum times too small and nothing anywhere raises.
    """
    engine = FakeEngine(accum=4)
    run = DistributedRun(model=engine, strategy="deepspeed", grad_accum=4, engine=engine)
    for _ in range(4):
        run.micro_step(FakeLoss([], value=1.0))
    assert engine.backwards == [1.0, 1.0, 1.0, 1.0], (
        "the engine must receive the unscaled loss; these values being 0.25 means the "
        "loss was scaled twice and the effective learning rate is wrong by 4x"
    )


def test_deepspeed_calls_step_every_micro_batch_because_the_engine_decides() -> None:
    engine = FakeEngine(accum=4)
    run = DistributedRun(model=engine, strategy="deepspeed", grad_accum=4, engine=engine)
    stepped = [run.micro_step(FakeLoss([])) for _ in range(8)]
    assert engine.steps == 8, (
        "DeepSpeed expects step() on every micro-batch and no-ops until the boundary; "
        "calling it only at the boundary skips the engine's own bookkeeping"
    )
    assert stepped == [False, False, False, True, False, False, False, True]


def test_the_two_strategies_report_their_scaling_contract() -> None:
    """A caller that needs to know can ask, instead of guessing from the strategy name."""
    assert DistributedRun(None, "deepspeed", 4, engine=FakeEngine(4)).scales_loss_internally
    assert not DistributedRun(None, "fsdp", 4).scales_loss_internally


def test_both_strategies_take_the_same_number_of_optimizer_steps() -> None:
    """The property the benchmark actually depends on, stated directly.

    If FSDP and DeepSpeed disagree about how many optimizer steps 12 micro-batches is,
    they are training on different global batches and the throughput comparison between
    them is meaningless, however fast either one runs.
    """
    accum, micro_batches = 4, 12
    opt = FakeOptimizer()
    fsdp = DistributedRun(None, "fsdp", accum, optimizer=opt)
    engine = FakeEngine(accum)
    ds = DistributedRun(engine, "deepspeed", accum, engine=engine)

    fsdp_boundaries = sum(fsdp.micro_step(FakeLoss([])) for _ in range(micro_batches))
    ds_boundaries = sum(ds.micro_step(FakeLoss([])) for _ in range(micro_batches))
    assert fsdp_boundaries == ds_boundaries == micro_batches // accum


def test_ray_is_refused_with_a_reason_rather_than_half_wired() -> None:
    from forge.training.distributed import build_run

    with pytest.raises(NotImplementedError, match="Ray Train"):
        build_run(DistConfig(strategy="ray", world_size=2, grad_accum=1), model=object())


# ---------------------------------------------------------------------------
# THE CONFIG GENERATOR EMITTED A CONFIG DEEPSPEED REJECTS.
#
# deepspeed_config() carried a WarmupDecayLR scheduler with warmup_min_lr, warmup_max_lr
# and warmup_num_steps. WarmupDecayLR also requires total_num_steps, so the first call to
# deepspeed.initialize() on a rented 2-GPU pod died with:
#
#   TypeError: WarmupDecayLR.__init__() missing 1 required positional argument:
#              'total_num_steps'
#
# Every existing test asserted the SHAPE of the returned dict: keys present, global batch
# derived correctly, zero_stage validated. None of them handed the dict to DeepSpeed,
# because that needs the library and a GPU. So a generator whose output the target refuses
# sat in the repo looking thoroughly tested.
#
# The lesson is narrow and worth keeping: testing that a function produces the dict you
# INTENDED is not testing that the dict is VALID. For anything whose output is consumed by
# another system, the contract worth pinning is that system's, not your own idea of it.
# ---------------------------------------------------------------------------

# WarmupDecayLR's required parameters, from DeepSpeed's own signature. Written down here
# so the check runs on a laptop with no deepspeed installed, and cross-checked against the
# real signature below whenever the library IS present.
WARMUP_DECAY_LR_REQUIRED = {"warmup_min_lr", "warmup_max_lr", "warmup_num_steps",
                            "total_num_steps"}


def test_no_scheduler_is_emitted_when_the_step_count_is_unknown() -> None:
    """An absent scheduler beats an invalid one.

    A throughput benchmark does not want a schedule: a decaying learning rate changes what
    the optimizer computes and changes nothing about how long a step takes.
    """
    from forge.training.distributed import deepspeed_config

    conf = deepspeed_config(DistConfig(strategy="deepspeed", world_size=2,
                                       per_device_batch=16, grad_accum=2), lr=2e-5)
    assert "scheduler" not in conf, (
        "emitting a scheduler without total_num_steps produces a config DeepSpeed refuses "
        "to construct; omitting it produces one that works"
    )


def test_an_emitted_scheduler_carries_every_parameter_deepspeed_requires() -> None:
    """THE REGRESSION. This is the assertion whose absence cost a pod session."""
    from forge.training.distributed import deepspeed_config

    conf = deepspeed_config(
        DistConfig(strategy="deepspeed", world_size=2, per_device_batch=16, grad_accum=2),
        lr=2e-5, warmup_steps=10, total_num_steps=100,
    )
    params = set(conf["scheduler"]["params"])
    missing = WARMUP_DECAY_LR_REQUIRED - params
    assert not missing, f"WarmupDecayLR would raise on construction, missing {missing}"


def test_a_schedule_that_warms_up_past_the_end_of_training_is_refused() -> None:
    """500 warmup steps inside a 100-step run is a config that runs and trains nothing:
    the learning rate never leaves the ramp. It fails here instead."""
    from forge.training.distributed import deepspeed_config

    with pytest.raises(ValueError, match="warm up past the end"):
        deepspeed_config(
            DistConfig(strategy="deepspeed", world_size=2, per_device_batch=16,
                       grad_accum=2),
            lr=2e-5, warmup_steps=500, total_num_steps=100,
        )


def test_the_required_parameter_list_matches_deepspeeds_actual_signature() -> None:
    """Keeps the hardcoded list honest wherever deepspeed is installed.

    Without this, WARMUP_DECAY_LR_REQUIRED is my belief about DeepSpeed's API rather than
    DeepSpeed's API, which is precisely the mistake being fixed. Skips cleanly on a
    machine without the [dist] extra, and runs on any machine that has it.
    """
    import inspect

    deepspeed = pytest.importorskip("deepspeed", reason="deepspeed is in the [dist] extra")
    from deepspeed.runtime.lr_schedules import WarmupDecayLR

    sig = inspect.signature(WarmupDecayLR.__init__)
    required = {
        name for name, param in sig.parameters.items()
        if name not in ("self", "optimizer") and param.default is inspect.Parameter.empty
    }
    assert required <= WARMUP_DECAY_LR_REQUIRED, (
        f"DeepSpeed {deepspeed.__version__} requires {required - WARMUP_DECAY_LR_REQUIRED} "
        "which this test's list does not include"
    )
