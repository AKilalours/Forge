from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOUR = ROOT / "docs/tour/index.html"


def test_tour_has_a_voice_story_chapter_and_links_to_the_live_app():
    page = TOUR.read_text()
    assert 'tag:"VOICE-STORY",name:"Voice Story"' in page
    assert "assets/voice-story-flow.svg" in page
    assert "https://panagramforge-cqzwwskdjhbfv6hxwppvkz.streamlit.app/" in page
    assert "Illustrative preview" in page
    assert "predates Voice Story" in page


def test_tour_chapter_grid_and_copy_cover_six_chapters():
    page = TOUR.read_text()
    assert "repeat(6,1fr)" in page
    assert "Voice Story" in page
    assert "research behind Forge" in page
