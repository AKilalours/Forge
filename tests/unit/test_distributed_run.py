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
