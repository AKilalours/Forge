"""Generator backends behind one interface.

Note the asymmetry with serving. vLLM belongs HERE, on the generation side, where the
workload is autoregressive decoding of hundreds of thousands of documents and
continuous batching genuinely helps. It does not belong on the detector's serving path,
which runs a bidirectional encoder with a fixed window and no KV cache. See
docs/jd_coverage.md.

Every backend must report a concrete `revision`. `require_pinned_revision` refuses to
run against an unpinned model, because "generated with Qwen 7B" is not a reproducible
statement: the upstream repo can move and the same config would then produce different
data under the same dataset version.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

UNPINNED_MARKERS = ("TODO_PIN_AT_FIRST_RUN", "main", "", None)


@dataclass(frozen=True)
class Decoding:
    temperature: float
    top_p: float
    max_new_tokens: int = 1024
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class UnpinnedRevisionError(RuntimeError):
    pass


def require_pinned_revision(family: str, revision: str | None) -> str:
    if revision in UNPINNED_MARKERS:
        raise UnpinnedRevisionError(
            f"generator family {family!r} has revision {revision!r}. Pin an exact repo "
            "revision in configs/generation/generators.yaml before generating. An "
            "unpinned model means this dataset version cannot be reproduced."
        )
    return revision  # type: ignore[return-value]


@runtime_checkable
class Generator(Protocol):
    family: str
    model_id: str
    revision: str

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]: ...

    def generate_many(self, prompts: list[str], decodings: list[Decoding]) -> list[str]:
        """Generate one completion per prompt, each with its OWN decoding parameters.

        WHY THIS EXISTS. The runner used to call generate([one_prompt], d) per document.
        That is a correct but unusable way to drive vLLM: a single 640-token completion
        from a 3B model decodes at roughly 60-100 tok/s, about 8 seconds per document, so
        60k documents would take over 130 hours. Continuous batching is the entire reason
        vLLM is on the generation side at all, and one-request-at-a-time never engages it.

        Decodings are per-prompt rather than shared because assign_decoding derives a
        distinct seed from each document id, so a batch cannot share one SamplingParams.
        """
        ...

    def close(self) -> None:
        """Release whatever the backend is holding. Must be safe to call twice."""
        ...


# How much context to reserve per request.
#
# vLLM sizes its KV cache from the model's ADVERTISED max_position_embeddings unless told
# otherwise. Phi-3.5-mini advertises 131072, which needs 48 GiB of KV cache; a 24 GB card
# has about 13 GiB free after weights, so the engine refuses to start:
#
#   ValueError: To serve at least one request with the model's max seq len (131072),
#   48.01 GiB KV cache is needed, which is larger than the available KV cache memory
#
# Mirrors never need that. A mirror prompt is the extracted attributes plus instructions,
# a few hundred tokens, and generation is capped by Decoding.max_new_tokens (640 in the
# minimal config). 4096 leaves generous headroom.
#
# This is not only a fix for the crash. Reserving 131k of context on a model that also
# fits would still cost throughput, because every block held for a context we never use
# is a block unavailable for batching other requests.
DEFAULT_MAX_MODEL_LEN = 4096


class ContextTooSmallError(ValueError):
    pass


class GpuNotFreeError(RuntimeError):
    pass


def require_free_gpu(min_free_gib: float = 18.0) -> None:
    """Refuse to build an engine when the card is already occupied.

    WHY. A vLLM engine reserves gpu_memory_utilization (0.9) of the card at startup and
    dies if that much is not free. Twice now a run has been launched while an earlier job
    still held the GPU, and the failure arrived several minutes in, as

        ValueError: Free memory on device (0.56/23.53 GiB) on startup is less than
        desired GPU memory utilization (0.9, 21.17 GiB)

    which names the symptom and not the cause. The two causes worth distinguishing are a
    stale job the operator forgot about, and close() having failed to release the previous
    family's engine, which would be a bug in this file. Say both, up front, in a second.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
    except Exception:
        # No GPU visible, or a torch build that cannot report. Not a reason to block.
        return
    free_gib = free / 1024**3
    if free_gib >= min_free_gib:
        return
    raise GpuNotFreeError(
        f"only {free_gib:.2f} GiB of {total / 1024**3:.2f} GiB is free on the GPU, and an "
        f"engine needs about {min_free_gib:.0f}. Either another job still holds the card "
        "(check `tmux ls` and `nvidia-smi`), or the previous family's engine was not "
        "released, which would be a bug in VLLMGenerator.close()."
    )


class VLLMGenerator:
    """Offline batch generation for open-weight families."""

    def __init__(
        self,
        family: str,
        model_id: str,
        revision: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    ) -> None:
        self.family = family
        self.model_id = model_id
        self.revision = require_pinned_revision(family, revision)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self._llm = None

    def _check_fits(self, decoding: Decoding) -> None:
        """Refuse a decoding whose output alone cannot fit the reserved context.

        Without this, asking for more new tokens than max_model_len silently truncates
        every generation, and the mirror validator would then reject them all as
        length-ratio failures: a confusing symptom two layers from its cause.
        """
        if decoding.max_new_tokens >= self.max_model_len:
            raise ContextTooSmallError(
                f"max_new_tokens={decoding.max_new_tokens} does not fit in "
                f"max_model_len={self.max_model_len} for family {self.family!r}. Raise "
                "max_model_len or lower max_new_tokens."
            )

    def _load(self):  # pragma: no cover - needs a GPU
        if self._llm is None:
            from vllm import LLM

            require_free_gpu()
            self._llm = LLM(
                model=self.model_id,
                revision=self.revision,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
            )
        return self._llm

    def _params(self, decoding: Decoding):  # pragma: no cover - needs vllm
        from vllm import SamplingParams

        self._check_fits(decoding)
        return SamplingParams(
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_tokens=decoding.max_new_tokens,
            seed=decoding.seed,
        )

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:  # pragma: no cover
        # Check BEFORE _load(). Python evaluates the callee before the arguments, so
        # folding the check into _params() would run it only after the engine had
        # already been constructed, which defeats the point of failing up front.
        self._check_fits(decoding)
        params = self._params(decoding)
        outs = self._load().generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def generate_many(self, prompts: list[str], decodings: list[Decoding]) -> list[str]:  # pragma: no cover
        if len(prompts) != len(decodings):
            raise ValueError(f"{len(prompts)} prompts but {len(decodings)} decodings")
        if not prompts:
            return []
        for d in decodings:
            self._check_fits(d)
        params = [self._params(d) for d in decodings]
        # use_tqdm=False: vLLM's per-request progress bar rewrites the line thousands of
        # times, which is unreadable in a tee'd log and makes the log ungreppable. The
        # runner prints its own batch-level progress instead.
        outs = self._load().generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def close(self) -> None:  # pragma: no cover - needs a GPU
        """Tear the engine down and give the GPU memory back.

        A vLLM engine reserves gpu_memory_utilization (0.9 by default) of the card at
        startup and holds it for its lifetime. The runner used to keep one engine per
        family alive simultaneously, so the second family's engine found 1 GiB free and
        refused to start. Dropping the reference is not enough; the allocator caches
        blocks, so the cache has to be emptied explicitly.
        """
        if self._llm is None:
            return
        llm, self._llm = self._llm, None
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            # Best effort. Version-dependent, and never a reason to fail a run.
            pass
        del llm
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


class TransformersGenerator:
    """Fallback for machines without vLLM, such as an Apple Silicon laptop.

    Much slower. Useful for generating a few thousand documents to validate the mirror
    prompt before committing GPU hours to the full 400k run.
    """

    def __init__(self, family: str, model_id: str, revision: str, device: str = "auto") -> None:
        self.family = family
        self.model_id = model_id
        self.revision = require_pinned_revision(family, revision)
        self.device = device
        self._pipe = None

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:  # pragma: no cover
        from transformers import pipeline

        if self._pipe is None:
            self._pipe = pipeline(
                "text-generation", model=self.model_id, revision=self.revision, device_map=self.device
            )
        outs = self._pipe(
            prompts,
            do_sample=not decoding.greedy,
            temperature=decoding.temperature or None,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            return_full_text=False,
        )
        return [o[0]["generated_text"] for o in outs]

    def generate_many(self, prompts: list[str], decodings: list[Decoding]) -> list[str]:  # pragma: no cover
        """No continuous batching here, so run them one at a time and keep the contract."""
        if len(prompts) != len(decodings):
            raise ValueError(f"{len(prompts)} prompts but {len(decodings)} decodings")
        return [self.generate([p], d)[0] for p, d in zip(prompts, decodings, strict=True)]

    def close(self) -> None:  # pragma: no cover
        self._pipe = None


class APIGenerator:
    """The held-out frontier family. Records the served model string per request, because
    an API model can change underneath a stable name."""

    def __init__(self, family: str, model_id: str, endpoint: str) -> None:
        self.family = family
        self.model_id = model_id
        self.endpoint = endpoint
        self.revision = "recorded_at_run_time"

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:  # pragma: no cover
        raise NotImplementedError("Phase 5: wire the held-out API family")

    def generate_many(self, prompts: list[str], decodings: list[Decoding]) -> list[str]:  # pragma: no cover
        raise NotImplementedError("Phase 5: wire the held-out API family")

    def close(self) -> None:  # pragma: no cover
        return None


class FakeGenerator:
    """Deterministic stand-in so the whole mirror pipeline is verifiable without a GPU.

    It produces text of roughly the requested length from the prompt's own attributes.
    It is NOT a model and its output is NOT training data. The runner records
    provider="fake" on every record so a fake-generated dataset can never be mistaken
    for a real one downstream.
    """

    family = "fake"
    model_id = "forge/fake-generator"
    revision = "v1"

    _FILLER = (
        "The account sets out the background before turning to the details that follow. "
        "Several considerations bear on the question, and they are taken in turn below. "
        "Evidence from the period is uneven, which limits how firmly any conclusion can "
        "be drawn. A number of practitioners disagreed with the prevailing approach. "
        "Later commentary revisited the matter without reaching a settled view. "
        "The arrangement persisted for some time before circumstances changed again. "
    )

    def __init__(self, family: str = "fake", model_id: str | None = None, revision: str = "v1") -> None:
        self.family = family
        self.model_id = model_id or f"forge/fake-{family}"
        self.revision = revision

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:
        out = []
        for p in prompts:
            target = _target_from_prompt(p)
            # jitter deterministically off the prompt hash so lengths vary like a real model
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            n = max(int(target * (0.85 + (h % 30) / 100.0)), 20)
            words = (self._FILLER * (n // 50 + 2)).split()[:n]
            # Emit paragraph breaks. A stand-in that returns one unbroken block cannot
            # exercise anything downstream that depends on document structure, and
            # splice construction would silently yield zero documents. Real generators
            # paragraph their output, and one that does not is a bug worth surfacing.
            per_para = max(n // (3 + (h % 3)), 25)
            paras = [" ".join(words[i : i + per_para]) for i in range(0, len(words), per_para)]
            out.append("\n\n".join(paras))
        return out

    def generate_many(self, prompts: list[str], decodings: list[Decoding]) -> list[str]:
        """Deterministic stand-in: output depends only on the prompt, so batching is trivial."""
        if len(prompts) != len(decodings):
            raise ValueError(f"{len(prompts)} prompts but {len(decodings)} decodings")
        return self.generate(prompts, decodings[0]) if prompts else []

    def close(self) -> None:
        return None


_TARGET_RE = None


def _target_from_prompt(prompt: str, default: int = 300) -> int:
    import re as _re

    m = _re.search(r"approximately (\d+) tokens", prompt)
    return int(m.group(1)) if m else default
