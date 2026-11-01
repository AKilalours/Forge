from __future__ import annotations

import sys

import pytest

from scripts import smoke_providers


def test_generated_pdf_is_a_valid_minimal_pdf_with_only_the_fictional_sample():
    pdf = smoke_providers.make_fictional_pdf()
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.endswith(b"%%EOF\n")
    assert b"xref\n0 6\n" in pdf
    assert smoke_providers.PDF_TEXT.encode("ascii") in pdf


def test_pangram_smoke_reports_metadata_without_window_text(monkeypatch):
    from forge.experience import providers

    sent = {}

    def fake_analyze(text, *, api_key):
        sent.update(text=text, api_key=api_key)
        return {
            "model": "pangram-test",
            "version": "v-test",
            "prediction": "mixed",
            "windows": [{"text": "provider returned passage text"}],
        }

    monkeypatch.setattr(providers, "pangram_analyze", fake_analyze)
    result = smoke_providers._run_pangram("secret-value")
    assert sent == {"text": smoke_providers.FICTIONAL_TEXT, "api_key": "secret-value"}
    assert result["status"] == "success"
    assert result["window_count"] == 1
    assert "provider returned passage text" not in repr(result)
    assert "secret-value" not in repr(result)


def test_reducto_smoke_reports_parse_shape_without_extracted_text(monkeypatch):
    from forge.experience import providers

    sent = {}

    def fake_parse(data, filename, *, api_key):
        sent.update(data=data, filename=filename, api_key=api_key)
        return {
            "text": "provider extracted fictional passage",
            "pages": 1,
            "chunks": [{"text": "provider extracted fictional passage"}],
        }

    monkeypatch.setattr(providers, "reducto_parse", fake_parse)
    result = smoke_providers._run_reducto("another-secret")
    assert sent["data"].startswith(b"%PDF-1.4")
    assert sent["filename"] == "forge-fictional-smoke.pdf"
    assert sent["api_key"] == "another-secret"
    assert result["status"] == "success"
    assert result["pages"] == 1 and result["text_extracted"]
    assert "provider extracted fictional passage" not in repr(result)
    assert "another-secret" not in repr(result)


def test_provider_failure_does_not_persist_exception_text(monkeypatch):
    from forge.experience import providers

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("sensitive provider response and test-secret")

    monkeypatch.setattr(providers, "pangram_analyze", fail_analysis)
    result = smoke_providers.run_smoke("pangram", "test-secret")
    assert result == {"status": "failure", "error_type": "RuntimeError"}
    assert "sensitive provider response" not in repr(result)
    assert "test-secret" not in repr(result)


def test_cli_confirmation_is_required_before_any_provider_call(monkeypatch, tmp_path):
    monkeypatch.setenv("PANGRAM_API_KEY", "configured-test-key")
    monkeypatch.setattr(sys, "argv", [
        "smoke_providers.py", "--provider", "pangram", "--confirm-external",
        "--out", str(tmp_path / "report.json"),
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: "NO")
    monkeypatch.setattr(
        smoke_providers, "_run_pangram",
        lambda _key: pytest.fail("provider call ran without the exact typed confirmation"),
    )
    with pytest.raises(SystemExit):
        smoke_providers.main()
    assert not (tmp_path / "report.json").exists()
