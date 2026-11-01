"""Build a private, human-authored keepsake from a passage and its context.

No model writes the author's story. The application only suggests exact excerpts from
the supplied text; the author chooses one and writes the meaning in their own words.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_EXCERPT_CHARS = 240


@dataclass(frozen=True)
class QuoteCandidate:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class VoiceCapsule:
    title: str
    author: str
    excerpt: str
    source_page: int | None
    detail: str
    intention: str
    keepsake: str
    source_name: str
    created_at: str

    def as_dict(self) -> dict:
        return asdict(self)


def quote_candidates(text: str, *, limit: int = 8) -> list[QuoteCandidate]:
    """Offer exact, bounded excerpts in document order; never paraphrase the source."""
    if limit < 1:
        raise ValueError("limit must be positive")
    candidates: list[QuoteCandidate] = []
    seen: set[str] = set()
    for raw in _SENTENCE.split(text):
        quote = " ".join(raw.split()).strip(" \t\"'“”‘’")
        if len(quote) < 28 or len(quote) > MAX_EXCERPT_CHARS:
            continue
        if quote.casefold() in seen:
            continue
        candidates.append(QuoteCandidate(quote))
        seen.add(quote.casefold())
        if len(candidates) == limit:
            break
    return candidates


def parsed_quote_candidates(parsed: dict, *, limit: int = 8) -> list[QuoteCandidate]:
    """Preserve Reducto page references for quotes extracted from parsed blocks."""
    candidates: list[QuoteCandidate] = []
    seen: set[str] = set()
    for chunk in parsed.get("chunks", []):
        blocks = chunk.get("blocks", [])
        page = next((b.get("page") for b in blocks if b.get("page") is not None), None)
        for quote in quote_candidates(chunk.get("text", ""), limit=limit):
            if quote.text.casefold() in seen:
                continue
            candidates.append(QuoteCandidate(quote.text, page))
            seen.add(quote.text.casefold())
            if len(candidates) == limit:
                return candidates
    return candidates


def capsule_from_answers(
    *,
    title: str,
    author: str,
    excerpt: str,
    source_page: int | None = None,
    detail: str,
    intention: str,
    keepsake: str,
    source_name: str = "Pasted text",
) -> VoiceCapsule:
    """Validate and package only the passage and reflections supplied by the author."""
    values = {
        "title": title,
        "excerpt": excerpt,
        "detail": detail,
        "intention": intention,
        "keepsake": keepsake,
    }
    missing = [key for key, value in values.items() if not value.strip()]
    if missing:
        raise ValueError(f"complete these fields first: {', '.join(missing)}")
    if len(excerpt) > MAX_EXCERPT_CHARS:
        raise ValueError(f"excerpt must be {MAX_EXCERPT_CHARS} characters or fewer")
    return VoiceCapsule(
        title=title.strip()[:100],
        author=author.strip()[:80],
        excerpt=excerpt.strip(),
        source_page=source_page,
        detail=detail.strip()[:1000],
        intention=intention.strip()[:1000],
        keepsake=keepsake.strip()[:1000],
        source_name=source_name.strip()[:120] or "Pasted text",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def render_capsule_html(capsule: VoiceCapsule) -> str:
    """Return a self-contained, offline HTML keepsake with optional browser narration."""
    e = html.escape
    author = e(capsule.author) if capsule.author else "A voice, kept"
    source = e(capsule.source_name)
    page = f" · page {capsule.source_page}" if capsule.source_page is not None else ""
    text_to_speak = " ".join((capsule.excerpt, capsule.detail, capsule.intention, capsule.keepsake))
    speech_json = (
        json.dumps(text_to_speak, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#101410"><title>{e(capsule.title)} · Voice Story</title>
<style>
:root{{--ink:#eee8dc;--muted:#b4b7a7;--paper:#171b16;--line:#455042;--glow:#d6b77c}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 50% 0,#354234 0,transparent 54%),#0b0f0c;color:var(--ink);font:17px/1.7 Georgia,serif}}
main{{width:min(780px,92vw);margin:7vh auto;padding:clamp(24px,7vw,72px);background:linear-gradient(145deg,#1b211b,#131713);border:1px solid var(--line);border-radius:28px;box-shadow:0 30px 100px #0008;animation:arrive .9s ease both}}
@keyframes arrive{{from{{opacity:0;transform:translateY(18px)}}to{{opacity:1;transform:none}}}}
.eyebrow{{font:600 11px/1.4 ui-sans-serif,system-ui;letter-spacing:.18em;text-transform:uppercase;color:var(--glow)}}h1{{font-size:clamp(38px,8vw,72px);line-height:1.03;letter-spacing:-.04em;font-weight:400;margin:20px 0 8px}}.by{{font:14px ui-sans-serif,system-ui;color:var(--muted)}}
blockquote{{margin:44px 0 12px;padding:0 0 0 24px;border-left:2px solid var(--glow);font-size:clamp(21px,4vw,31px);line-height:1.45}}.citation{{font:12px ui-sans-serif,system-ui;color:var(--muted)}}
section{{padding-top:25px;margin-top:28px;border-top:1px solid #ffffff18}}h2{{font:600 11px ui-sans-serif,system-ui;letter-spacing:.15em;text-transform:uppercase;color:var(--glow)}}.answer{{white-space:pre-wrap}}
.listen{{display:inline-flex;align-items:center;gap:10px;margin-top:36px;padding:12px 18px;border:1px solid var(--line);border-radius:999px;background:#20271f;color:var(--ink);font:600 13px ui-sans-serif,system-ui;cursor:pointer}}.listen:focus-visible{{outline:3px solid var(--glow);outline-offset:3px}}
footer{{margin-top:32px;font:12px/1.6 ui-sans-serif,system-ui;color:var(--muted)}}.wave{{display:flex;gap:4px;height:20px;align-items:center}}.wave i{{width:2px;height:6px;background:var(--glow);animation:pulse 1.2s ease-in-out infinite alternate}}.wave i:nth-child(3n){{animation-delay:.3s}}.wave i:nth-child(2n){{animation-delay:.6s}}@keyframes pulse{{to{{height:18px}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{animation:none!important;scroll-behavior:auto!important}}}}
</style></head><body><main>
<div class="eyebrow">A voice, kept</div><h1>{e(capsule.title)}</h1><div class="by">By {author}</div>
<blockquote>{e(capsule.excerpt)}</blockquote><div class="citation">From {source}{e(page)}</div>
<section><h2>The detail I remember</h2><div class="answer">{e(capsule.detail)}</div></section>
<section><h2>What I meant</h2><div class="answer">{e(capsule.intention)}</div></section>
<section><h2>For whoever reads this later</h2><div class="answer">{e(capsule.keepsake)}</div></section>
<button class="listen" type="button" id="listen" aria-pressed="false"><span class="wave" aria-hidden="true">{''.join('<i></i>' for _ in range(9))}</span><span>Listen to this story</span></button>
<footer>Made from an excerpt you chose and reflections you wrote. Detector signals are not included as a judgment of the author. This keepsake is stored nowhere by Forge.</footer>
</main><script>
const story={speech_json};const button=document.getElementById('listen');let speaking=false;
button.addEventListener('click',()=>{{if(!('speechSynthesis'in window)){{button.querySelector('span:last-child').textContent='Speech is not available in this browser';return}}if(speaking){{speechSynthesis.cancel();speaking=false;button.setAttribute('aria-pressed','false');button.querySelector('span:last-child').textContent='Listen to this story';return}}const line=new SpeechSynthesisUtterance(story);line.rate=.88;line.onend=()=>{{speaking=false;button.setAttribute('aria-pressed','false');button.querySelector('span:last-child').textContent='Listen again'}};speaking=true;button.setAttribute('aria-pressed','true');button.querySelector('span:last-child').textContent='Stop narration';speechSynthesis.speak(line)}});
</script></body></html>"""
