"""Dynamic batching.

The tradeoff is explicit: waiting adds latency to every request and multiplies
throughput. The wait window is tuned against the P95 latency budget in the release gate,
not chosen by feel.

The queue policy below is FIFO on purpose. Sorting by length to reduce padding waste is
the obvious optimisation and it starves long documents indefinitely under load, which
shows up as a P99 latency cliff on exactly the requests users care most about. FORGE
windows every document to a fixed 512 tokens anyway, so padding waste is bounded and the
optimisation buys little.

`plan_batches` is pure and tested. The server loop that consumes it is not, because it
needs a live model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BatchPolicy:
    # 8, not 32, and measured. See reports/experiments/inference/cpu_latency.json: on the
    # CPU path FORGE deploys on, throughput is flat within 13% from batch 1 to 16 and gets
    # WORSE at 32. A dynamic batcher whose max_batch sits past the point where batching
    # stops paying adds latency to every request and buys nothing back.
    max_batch: int = 8
    max_wait_ms: int = 10
    max_queue: int = 1000

    def __post_init__(self) -> None:
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")


@dataclass
class QueuedRequest:
    request_id: str
    n_windows: int          # a long document is several windows, so it costs several slots
    enqueued_ms: float


@dataclass
class PlannedBatch:
    requests: list[QueuedRequest] = field(default_factory=list)
    n_windows: int = 0

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]


class QueueOverflow(RuntimeError):
    pass


def plan_batches(queue: list[QueuedRequest], policy: BatchPolicy, now_ms: float) -> list[PlannedBatch]:
    """Pack the queue into batches by WINDOW count, preserving arrival order.

    Batching by request count rather than window count is the trap: thirty-two
    single-window requests and thirty-two hundred-window requests are the same batch size
    and wildly different amounts of work, so the batch that happens to contain long
    documents blows the latency budget.

    A single request larger than max_batch gets its own batch rather than being dropped.
    """
    if len(queue) > policy.max_queue:
        raise QueueOverflow(
            f"{len(queue)} requests queued, limit is {policy.max_queue}. Shed load and "
            "return 503 rather than growing an unbounded queue behind a latency budget."
        )
    batches: list[PlannedBatch] = []
    current = PlannedBatch()
    for req in queue:
        if req.n_windows > policy.max_batch:
            if current.requests:
                batches.append(current)
                current = PlannedBatch()
            batches.append(PlannedBatch([req], req.n_windows))
            continue
        if current.n_windows + req.n_windows > policy.max_batch:
            batches.append(current)
            current = PlannedBatch()
        current.requests.append(req)
        current.n_windows += req.n_windows
    if current.requests:
        batches.append(current)
    return batches


def should_flush(queue: list[QueuedRequest], policy: BatchPolicy, now_ms: float) -> bool:
    """Flush when the batch is full OR the oldest request has waited long enough.

    The age check is what bounds tail latency. Without it a quiet period leaves a single
    request sitting in the queue until enough traffic arrives to fill a batch, and that
    request sees seconds of latency at low load, which is the opposite of what anyone
    expects.
    """
    if not queue:
        return False
    total = sum(r.n_windows for r in queue)
    if total >= policy.max_batch:
        return True
    return (now_ms - min(r.enqueued_ms for r in queue)) >= policy.max_wait_ms


def added_latency_ms(policy: BatchPolicy) -> int:
    """Worst-case latency this policy adds to a single request."""
    return policy.max_wait_ms


def fits_latency_budget(policy: BatchPolicy, model_p95_ms: float, budget_p95_ms: float) -> bool:
    return added_latency_ms(policy) + model_p95_ms <= budget_p95_ms
