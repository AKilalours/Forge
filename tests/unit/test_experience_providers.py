from __future__ import annotations

from pathlib import Path

import pytest

from forge.experience.providers import pangram_analyze, reducto_parse


class FakePangram:
    def __init__(self, *, api_key):
        assert api_key == "test-key"
        self.request = None

    def list_models(self):
        return ["pangram-4"]

    def predict(self, text, *, model, public_dashboard_link):
        self.request = (text, model, public_dashboard_link)
        return {
            "stage": "STAGE_SUCCESS",
            "version": "4.0",
            "prediction_short": "mixed",
            "fraction_ai": 0.1,
            "fraction_ai_assisted": 0.2,
            "fraction_human": 0.7,
            "windows": [{"label": "human", "ai_assistance_score": 0.1}],
        }


def test_pangram_uses_available_model_and_disables_public_dashboard():
    client = FakePangram(api_key="test-key")
    result = pangram_analyze("A personal paragraph.", api_key="test-key", client_factory=lambda **kw: client)
    assert result["model"] == "pangram-4"
    assert result["prediction"] == "mixed"
    assert client.request == ("A personal paragraph.", "pangram-4", False)


def test_pangram_rejects_oversized_text_before_provider_call():
    with pytest.raises(ValueError, match="characters"):
        pangram_analyze("x" * 50_001, api_key="test-key", client_factory=FakePangram)


def test_reducto_upload_is_removed_after_parse():
    uploaded_paths: list[Path] = []

    class FakeReducto:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.upload = self
            self.parse = self
            self.file_id = "file-123"
            self.result = {"chunks": [{
                "content": "A sentence extracted from the letter.",
                "blocks": [{"type": "Text", "content": "A sentence extracted from the letter.",
                            "bbox": {"page": 2}}],
            }]}
            self.usage = {"num_pages": 2}
            self.job_id = "job-123"

        def __call__(self, **kwargs):
            if "file" in kwargs:
                path = kwargs["file"]
                assert path.exists()
                uploaded_paths.append(path)
                return self
            return self

        def run(self, *, input):
            assert input == "file-123"
            return self

    client = FakeReducto(api_key="test-key")

    def factory(**kwargs):
        assert kwargs["api_key"] == "test-key"

        class Adapter:
            def __init__(self):
                self.parse = client

            def upload(self, **kwargs):
                return client(**kwargs)

        return Adapter()

    parsed = reducto_parse(b"%PDF-test", "letter.pdf", api_key="test-key", client_factory=factory)
    assert parsed["pages"] == 2
    assert parsed["chunks"][0]["blocks"][0]["page"] == 2
    assert uploaded_paths and not uploaded_paths[0].exists()


def test_reducto_rejects_unsupported_file_before_provider_call():
    with pytest.raises(ValueError, match="accepts"):
        reducto_parse(b"no", "letter.exe", api_key="test-key")
