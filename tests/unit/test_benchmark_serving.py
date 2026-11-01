from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_serving import percentile, validate_base_url


def test_nearest_rank_percentile_uses_observed_values_only():
    assert percentile([9.0, 1.0, 5.0, 3.0, 7.0], 0.50) == 5.0
    assert percentile([9.0, 1.0, 5.0, 3.0, 7.0], 0.95) == 9.0


@pytest.mark.parametrize("q", [0.0, 1.01])
def test_percentile_rejects_invalid_quantiles(q):
    with pytest.raises(ValueError):
        percentile([1.0], q)


def test_remote_service_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="refusing a non-local target"):
        validate_base_url("https://example.org", allow_remote=False)
    assert validate_base_url("https://example.org/", allow_remote=True) == "https://example.org"


@pytest.mark.parametrize(
    "url",
    ["ftp://localhost", "http://user:pass@localhost", "http://localhost/path", "http://localhost/?key=x"],
)
def test_service_url_rejects_non_http_or_embedded_request_state(url):
    with pytest.raises(ValueError):
        validate_base_url(url, allow_remote=True)


def test_localhost_is_the_default_allowed_target():
    assert validate_base_url("http://127.0.0.1:8000", allow_remote=False) == "http://127.0.0.1:8000"


def test_committed_http_benchmark_is_complete_and_contains_no_text():
    artifact_path = Path(__file__).resolve().parents[2] / "reports/experiments/serving/local_cpu_http.json"
    artifact_text = artifact_path.read_text()
    artifact = json.loads(artifact_text)
    assert artifact["target_origin"].startswith("http://127.0.0.1:")
    assert artifact["workload"]["samples_per_level"] == 20
    assert [row["concurrency"] for row in artifact["levels"]] == [1, 2, 4]
    assert all(row["http_statuses"] == {"200": 20} and not row["errors"]
               for row in artifact["levels"])
    assert "The blue cup stayed" not in artifact_text
