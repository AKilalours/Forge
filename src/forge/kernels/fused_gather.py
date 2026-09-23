"""Python side of the fused row gather, with the reference it must agree with.

The kernel exists because the training profile named it: `aten::scatter_add_` is 20.87% of
device time at the shapes this project trains at, against 4.47% for every matrix multiply
combined. See kernels/cuda/fused_gather.cu for what it does differently.

THREE THINGS THIS MODULE IS CAREFUL ABOUT.

It compiles on first use and never at import. A CUDA extension takes tens of seconds to
build, and an import that does that turns every CLI invocation, including `--help` on a
CPU box, into a build. `load()` caches in torch's extension directory, so the cost is paid
once per machine.

It degrades to the reference instead of failing. On a machine with no CUDA, no nvcc, or a
toolchain that cannot build the extension, `fused_gather` returns the pure-torch path and
`backend()` says which one ran. A kernel that makes the project unrunnable on a laptop is
not an optimisation, and a benchmark that silently compared torch against torch would be
worse than no benchmark, which is why the backend is reported rather than inferred.

The reference is `table[index]`, not a hand-written equivalent. Its backward IS
`aten::scatter_add_`, so the comparison is against the exact op the profile measured.
"""

from __future__ import annotations

import functools
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parents[3] / "kernels" / "cuda"
SOURCE = KERNEL_DIR / "fused_gather.cu"


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile and cache the extension, or return the reason it is unavailable."""
    import torch

    if not torch.cuda.is_available():
        return None, "no CUDA device"
    if not SOURCE.exists():
        return None, f"missing {SOURCE}"
    try:
        from torch.utils.cpp_extension import load

        module = load(
            name="forge_fused_gather",
            sources=[str(SOURCE)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return module, None
    except Exception as error:                  # noqa: BLE001 - reported, never swallowed
        return None, f"{type(error).__name__}: {error}"


def available() -> bool:
    return _extension()[0] is not None


def unavailable_reason() -> str | None:
    return _extension()[1]


def backend() -> str:
    """"cuda" or "torch". Printed by the benchmark so a result cannot misattribute itself."""
    return "cuda" if available() else "torch"


def reference_gather(table, index):
    """The op the profile measured: advanced indexing, whose backward is scatter_add_."""
    return table[index]


def _make_autograd():
    import torch

    class FusedGatherRows(torch.autograd.Function):
        @staticmethod
        def forward(ctx, table, index):
            module, _ = _extension()
            ctx.save_for_backward(index)
            ctx.table_rows = table.size(0)
            ctx.table_requires_grad = table.requires_grad
            return module.gather_rows(table.contiguous(), index.contiguous())

        @staticmethod
        def backward(ctx, grad_out):
            module, _ = _extension()
            (index,) = ctx.saved_tensors
            if not ctx.table_requires_grad:
                return None, None
            grad_table = module.scatter_add_rows(
                grad_out.contiguous(), index, ctx.table_rows
            )
            return grad_table, None

    return FusedGatherRows


@functools.lru_cache(maxsize=1)
def _autograd():
    return _make_autograd()


def fused_gather(table, index):
    """Gather whole rows of `table` by `index`, on the fused kernel when it is available.

    Falls back to `table[index]`, which is numerically the same op, so a caller never has
    to branch on the backend. Check `backend()` if you need to know which one ran.
    """
    if not available():
        return reference_gather(table, index)
    return _autograd().apply(table, index)
