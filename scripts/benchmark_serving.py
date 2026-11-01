"""Measure the running FastAPI text endpoint end to end under bounded concurrency.

This complements ``benchmark_inference.py``, which times model forwards only. It sends a
fixed, fictional sample and records first-request startup/model-load time separately from
warm request latency. By default the target must be loopback so an accidental invocation
cannot load-test a public service. Use ``--allow-remote`` only for an explicitly approved
staging target. The JSON report contains no request text or response body.

Example:
    python -m uvicorn api.forge_app:app --host 127.0.0.1 --port 8000
    python scripts/benchmark_serving.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import platform
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/experiments/serving"
SAMPLE_TEXT = (
    "The blue cup stayed beside the window through every Sunday afternoon. "
    "Its chipped rim caught the last warm light, and the small ritual made an empty "
    "chair feel like a place where a conversation could continue."
)
CONCURRENCY_LEVELS = (1, 2, 4)
MAX_CONCURRENCY = 128
MAX_SAMPLES = 5_000


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank quantile; it does not interpolate nonexistent observations."""
    if not values or not 0 < q <= 1:
        raise ValueError("percentile requires samples and q in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def validate_base_url(base_url: str, *, allow_remote: bool) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an http(s) URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL cannot embed credentials, query parameters, or fragments")
    if parsed.path not in {"", "/"}:
        raise ValueError("pass only the service origin; the script appends the text endpoint")
    if not allow_remote and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("refusing a non-local target; pass --allow-remote for approved staging")
    return base_url.rstrip("/")


def parse_concurrency_levels(raw: str) -> tuple[int, ...]:
    """Parse a bounded, duplicate-free comma-separated concurrency schedule."""
    try:
        levels = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("concurrency levels must be comma-separated integers") from exc
    if not levels or any(level < 1 or level > MAX_CONCURRENCY for level in levels):
        raise ValueError(f"concurrency levels must be between 1 and {MAX_CONCURRENCY}")
    if len(set(levels)) != len(levels):
        raise ValueError("concurrency levels cannot contain duplicates")
    return levels


def request_once(endpoint: str, timeout_s: float) -> tuple[float, int, str | None]:
    body = json.dumps({"text": SAMPLE_TEXT}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            error = None if 200 <= status < 300 else f"http_{status}"
            if error is None:
                try:
                    payload = json.loads(response.read(2_000_000))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    error = "invalid_json"
                else:
                    if not isinstance(payload, dict) or not {"available", "arms", "unavailable"} <= payload.keys():
                        error = "invalid_response_schema"
                    elif not payload["available"] or not payload["arms"]:
                        error = "no_model_arm_scored"
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"http_{status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status = 0
        error = type(exc).__name__
    return (time.perf_counter() - start) * 1000, status, error


def measure_level(endpoint: str, samples: int, concurrency: int, timeout_s: float) -> dict[str, Any]:
    # One warmup per worker; retain its outcomes in the warm-up record rather than the
    # latency distribution. The separate first request is reported as cold_start_ms.
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_once, endpoint, timeout_s) for _ in range(concurrency)]
        warmups = [future.result() for future in futures]
        start = time.perf_counter()
        futures = [pool.submit(request_once, endpoint, timeout_s) for _ in range(samples)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        wall_s = time.perf_counter() - start

    latencies = [row[0] for row in results]
    errors = Counter(error for _, _, error in results if error)
    return {
        "concurrency": concurrency,
        "requests": samples,
        "wall_seconds": round(wall_s, 4),
        "requests_per_second": round(samples / wall_s, 3) if wall_s else None,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "http_statuses": dict(Counter(status for _, status, _ in results)),
        "errors": dict(errors),
        "warmup_errors": sum(1 for _, _, error in warmups if error),
    }


def run(
    base_url: str,
    *,
    samples: int,
    concurrency_levels: tuple[int, ...] = CONCURRENCY_LEVELS,
    timeout_s: float,
    allow_remote: bool,
) -> dict[str, Any]:
    origin = validate_base_url(base_url, allow_remote=allow_remote)
    endpoint = f"{origin}/v1/text/analyze"
    cold_ms, cold_status, cold_error = request_once(endpoint, timeout_s)
    levels = []
    for concurrency in concurrency_levels:
        result = measure_level(endpoint, samples, concurrency, timeout_s)
        levels.append(result)
        print(
            f"concurrency={concurrency} rps={result['requests_per_second']} "
            f"p50/p95/p99={result['latency_ms']['p50']}/"
            f"{result['latency_ms']['p95']}/{result['latency_ms']['p99']}ms "
            f"errors={result['errors']}",
            flush=True,
        )
    return {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "target_origin": origin,
        "endpoint": "/v1/text/analyze",
        "client": {"python": platform.python_version(), "platform": platform.platform()},
        "workload": {
            "kind": "fixed fictional text; never stored in this artifact",
            "characters": len(SAMPLE_TEXT),
            "concurrency_levels": list(concurrency_levels),
            "samples_per_level": samples,
            "timeout_seconds": timeout_s,
        },
        "first_request": {
            "latency_ms": round(cold_ms, 2),
            "http_status": cold_status,
            "error": cold_error,
            "note": "Includes any lazy model loading triggered by the first text request.",
        },
        "levels": levels,
        "scope": (
            "End-to-end HTTP response time from this client to the configured service. "
            "This is not a multi-host capacity test, a statistically powered SLO study, "
            "or a price measurement. Report target hardware and deployment separately."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=30,
                        help="measured requests per concurrency level (20–5000)")
    parser.add_argument("--concurrency", default=",".join(map(str, CONCURRENCY_LEVELS)),
                        help="comma-separated worker levels (default: 1,2,4; each 1–128)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--allow-remote", action="store_true",
                        help="permit the fixed fictional sample to reach a non-loopback URL")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON path; defaults to reports/experiments/serving/<timestamp>.json")
    args = parser.parse_args()
    if not 20 <= args.samples <= MAX_SAMPLES:
        parser.error(f"--samples must be between 20 and {MAX_SAMPLES} to keep runs bounded")
    if not 0.1 <= args.timeout <= 300:
        parser.error("--timeout must be between 0.1 and 300 seconds")
    try:
        concurrency_levels = parse_concurrency_levels(args.concurrency)
    except ValueError as exc:
        parser.error(str(exc))
    result = run(args.base_url, samples=args.samples, timeout_s=args.timeout,
                 allow_remote=args.allow_remote, concurrency_levels=concurrency_levels)
    out = args.out or DEFAULT_OUT / f"http-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"cold first request: {result['first_request']['latency_ms']}ms "
          f"HTTP {result['first_request']['http_status']}")
    print(f"artifact: {out}")
    if result["first_request"]["error"] or any(row["errors"] for row in result["levels"]):
        raise SystemExit("serving benchmark completed with request failures; inspect artifact")


if __name__ == "__main__":
    main()
