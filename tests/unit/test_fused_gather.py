"""The fused kernel must produce the gradients ATen produces. Speed is the second question.

A kernel that is faster and slightly wrong is the worst outcome available here: training
would still converge to something, the loss curve would look plausible, and every number
this project publishes would describe a model trained with a subtly wrong gradient on the
relative-position table. So correctness is tested first, against `table[index]`, whose
backward IS the `aten::scatter_add_` the profile measured.

ON EXACT EQUALITY. The forward is a pure copy and must match bitwise. The backward must
not: it accumulates with atomicAdd, so when several rows target one table row the summation
order is whatever the scheduler produced, and float addition is not associative. ATen's
scatter_add_ on CUDA is nondeterministic for the same reason, which is why
torch.use_deterministic_algorithms raises on it. Comparing with a tolerance is therefore
the correct comparison, not a concession.

Everything here skips without a CUDA device, which is most machines this repo runs on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from forge.kernels.fused_gather import (  # noqa: E402
    available,
    backend,
    fused_gather,
    reference_gather,
    unavailable_reason,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# The shapes the profile was taken at: DeBERTa-v3-base uses 512 relative position buckets
# and a 768-wide hidden size, and the gathers run per head over the window.
TABLE_ROWS, DIM = 512, 768


def _inputs(n_rows: int, dtype, duplicates: bool = True):
    generator = torch.Generator(device="cuda").manual_seed(0)
    table = torch.randn(TABLE_ROWS, DIM, device="cuda", dtype=dtype, generator=generator)
    high = TABLE_ROWS // 8 if duplicates else TABLE_ROWS
    index = torch.randint(0, high, (n_rows,), device="cuda", generator=generator)
    return table, index


@cuda
def test_the_extension_builds():
    # If it does not, every test below would silently exercise the torch fallback and pass
    # while testing nothing.
    assert available(), f"extension unavailable: {unavailable_reason()}"
    assert backend() == "cuda"


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_forward_is_bitwise_identical(dtype):
    table, index = _inputs(4096, dtype)

    assert torch.equal(fused_gather(table, index), reference_gather(table, index))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_gradient_matches_aten_with_duplicate_indices(dtype):
    """Duplicates are the whole point: without them no accumulation happens at all.

    DeBERTa's relative position buckets are shared across the window by construction, so
    every real call has heavy duplication. An index with no repeats would test a copy.
    """
    table, index = _inputs(4096, dtype, duplicates=True)
    a = table.clone().requires_grad_(True)
    b = table.clone().requires_grad_(True)
    upstream = torch.randn(4096, DIM, device="cuda", dtype=dtype)

    fused_gather(a, index).backward(upstream)
    reference_gather(b, index).backward(upstream)

    tolerance = 3e-3 if dtype is torch.bfloat16 else 1e-5
    assert torch.allclose(a.grad, b.grad, rtol=tolerance, atol=tolerance)


@cuda
def test_rows_never_touched_get_a_zero_gradient():
    # grad_table is allocated with zeros(); an empty() would leave whatever was in that
    # memory, and unused buckets would receive garbage gradients that training would then
    # apply. The failure would look like slow divergence, not like a bug.
    table, index = _inputs(64, torch.float32, duplicates=True)
    table = table.requires_grad_(True)

    fused_gather(table, index).backward(torch.ones(64, DIM, device="cuda"))

    untouched = torch.ones(TABLE_ROWS, dtype=torch.bool, device="cuda")
    untouched[index] = False
    assert table.grad[untouched].abs().max().item() == 0.0


@cuda
def test_an_empty_index_is_not_an_error():
    table, _ = _inputs(1, torch.float32)
    empty = torch.empty(0, dtype=torch.long, device="cuda")

    assert fused_gather(table, empty).shape == (0, DIM)


def test_the_fallback_is_the_same_op_on_a_machine_without_cuda():
    """The CPU path must stay usable, and must not pretend to be the kernel."""
    if torch.cuda.is_available():
        pytest.skip("this asserts the no-CUDA behaviour")
    table = torch.randn(8, 4, requires_grad=True)
    index = torch.tensor([0, 0, 3])

    out = fused_gather(table, index)
    out.backward(torch.ones_like(out))

    assert backend() == "torch"
    assert table.grad[0].allclose(torch.full((4,), 2.0))
