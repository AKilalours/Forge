"""FORGE on Streamlit Community Cloud. The same detectors, a different shell.

WHY THIS FILE EXISTS. The interface this project was built around is a FastAPI app in
`api/forge_app.py`, and it is still the reference. It cannot be deployed for free any more:
Hugging Face now requires a PRO subscription for Docker Spaces, and every other free host
either caps memory below one model or requires a payment card. Streamlit Community Cloud is
the one that does not, so this file is the interface layer rewritten for it.

Nothing about the detection changes. The verdict, the thresholds, the polarity resolution
and the occlusion attribution all come from `forge.*` exactly as they do in the FastAPI app.
If this page and that one ever disagree, this page is wrong.

MEMORY IS THE DESIGN CONSTRAINT. The free tier is about 2.7 GB of RAM. One text arm is a
float32 DeBERTa-v3-base at roughly 740 MB resident, and the FastAPI page loads both at once
to compare them. Two arms plus torch plus the image detector does not fit, and the failure
mode is the container being killed mid-request, which a reader cannot interpret. So this
page holds ONE arm at a time and says so, and switching arms evicts the other rather than
quietly accumulating. The two-arm comparison is the experiment's result and it is reported
in docs/evaluation.md, where it belongs; a live page is not the place that finding lives.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

WEIGHTS_REPO = os.getenv("FORGE_WEIGHTS_REPO", "Akilalourdes/forge-detect-weights")
EXPERIMENT = {"baseline": "forge_min_baseline", "mirror": "forge_min_mirror"}

st.set_page_config(page_title="FORGE Detect", page_icon="◈", layout="wide")


# ------------------------------------------------------------------ weights on demand

@st.cache_resource(show_spinner=False)
def ensure_weights(arm: str) -> str | None:
    """Fetch one arm's checkpoint from the Hub into the path the loader expects.

    ONE ARM, NOT BOTH. Disk on this tier is as tight as memory, and each checkpoint is
    735 MB. Fetching on demand means a visitor who only uses the image tab never pays for
    either. Returns None on success, or a reason a reader can act on.
    """
    from huggingface_hub import hf_hub_download

    out = pathlib.Path("outputs") / EXPERIMENT[arm]
    if (out / "best.pt").exists() and (out / "summary.json").exists():
        return None
    try:
        for name in ("best.pt", "summary.json"):
            hf_hub_download(
                repo_id=WEIGHTS_REPO,
                filename=f"{EXPERIMENT[arm]}/{name}",
                local_dir="outputs",
            )
    except Exception as error:  # noqa: BLE001 - shown to the reader, never swallowed
        return f"{type(error).__name__}: {error}"
    return None


# ------------------------------------------------------------------------- presentation

# THE SAME CARDS AS THE FASTAPI PAGE. `forge.ui.render` builds the identical markup against
# the identical stylesheet in `forge.ui.theme`, which api/forge_app.py also imports. The two
# shells differ only in what collects the input: a browser fetch there, a Streamlit widget
# here. Rendering the results twice from two hand-written templates is exactly how this
# project has produced pages that disagree with themselves, so there is one template.
import streamlit.components.v1 as components  # noqa: E402


def show(body: str, height: int) -> None:
    from forge.ui.render import document

    components.html(document(body), height=height, scrolling=True)


# ------------------------------------------------------------------------------ text tab

def text_tab() -> None:
    text = st.text_area(
        "Paste text to analyse",
        height=300,
        label_visibility="collapsed",
        placeholder="Paste text to analyse. A few paragraphs works best; very short "
                    "passages carry little signal.",
    )
    # THE ARM SELECTOR IS NOT A QUESTION FOR A VISITOR. It was the third thing on the page,
    # asking someone to choose between "mirror" and "baseline" with no basis for deciding
    # and no idea what the words mean. The two arms are this project's experimental
    # control, not a product feature: B is what is deployed and what a visitor should get
    # without being asked. Anyone who wants the comparison opens the section below and
    # finds it explained in two sentences.
    LABELS = {"B: matched mirrors": "mirror", "A: random synthetic": "baseline"}
    arm = "mirror"
    with st.expander("Compare the two detectors"):
        st.markdown(
            "This project trained **two** detectors on the same human writing but "
            "different synthetic writing.\n\n"
            "- **A** learned from synthetic text generated at random.\n"
            "- **B** learned from synthetic text written to mirror specific human "
            "documents, which is the idea being tested: that choosing *which* synthetic "
            "data to generate beats generating more of it.\n\n"
            "On text like this they agree, because both were trained for it. The "
            "experiment is about where they come apart, and that is reported in "
            "`docs/evaluation.md`, not here. **B serves your result** unless you change it."
        )
        arm = LABELS[st.radio("Detector", list(LABELS), horizontal=True)]
    if not st.button("Analyse text", type="primary"):
        return
    if not text.strip():
        st.error("Paste some text first.")
        return

    from forge.inference.text_api import analyse
    from forge.ui.render import text_result

    # ONE ARM, which is what this module's docstring has always said this page does. It
    # previously loaded both: 701 MB each in float32, plus torch, plus the page, against a
    # host that allows about 2.7 GB. That does not fail cleanly, it gets the container
    # killed mid-request, and the visitor sees "Oh no. Error running app." with no reason.
    # The two-arm comparison IS the experiment's result, and it is reported in
    # docs/evaluation.md, where a measured comparison belongs. A live page is not where
    # that finding lives.
    with st.spinner(f"Loading the {arm} arm and scoring. The first run fetches its weights."):
        failure = ensure_weights(arm)
        if failure:
            st.error(f"could not fetch the {arm} checkpoint: {failure}")
            return
        # The scorer's shared cache serves the FastAPI comparison path and can hold both
        # arms. Streamlit's small-host path scores one arm at a time, so evict the prior
        # selection when the user switches instead of retaining two 700+ MB models.
        from forge.inference.scorer import load_arm

        previous_arm = st.session_state.get("_forge_loaded_arm")
        if previous_arm != arm:
            load_arm.cache_clear()
            st.session_state["_forge_loaded_arm"] = arm
        payload = analyse(text, arms=(arm,))

    show(text_result(payload), height=1180)


# ----------------------------------------------------------------------------- image tab

def image_tab() -> None:
    upload = st.file_uploader(
        "Drop an image, or click to choose one",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp", "gif", "mpo"],
        help="JPEG, PNG, WEBP, TIFF, BMP, GIF, MPO. Analysed in memory, never written to disk.",
    )
    if upload is None:
        return

    from forge.image.analysis import UnsupportedImage, analyse
    from forge.ui.render import image_result

    data = upload.read()
    # The SAME analysis the FastAPI route runs, from forge.image.analysis: forensics,
    # detector, robustness over eleven edits, occlusion attribution and the pixel maps. It
    # is the slowest thing on this page at roughly fifteen seconds on two CPUs, and that is
    # the honest cost of computing every panel rather than asserting one.
    with st.spinner("Analysing. Forensics, detector, robustness over 11 edits, attribution."):
        try:
            # Occlusion attribution and the pixel maps stay: they are what SHOW the
            # analysis rather than assert it. Detector robustness does not: eleven
            # re-scored transforms cost 6.5 s of a 14 s analysis and render as a row of
            # chips a visitor has no way to act on. It remains on the FastAPI page.
            payload = analyse(data, filename=upload.name, with_stability=False,
                              with_robustness=False)
        except UnsupportedImage as refusal:
            st.error(str(refusal))
            return

    # The embed is an iframe with a fixed height, so it is sized from what actually
    # rendered rather than one guessed number: a page cut off mid-card reads as a bug.
    height = 1150
    if payload.get("robustness"):
        height += 260
    if payload.get("attribution_map"):
        height += 620
    if payload.get("maps"):
        height += 620
    # TALL ENOUGH FOR THE PANELS IT RENDERS. At 560 the occlusion map and the pixel
    # statistics were below the cut and looked as though they had not been computed,
    # which is exactly the complaint that sent me looking. components.html crops to this
    # height; it does not grow to fit.
    show(image_result(payload, compact=True), height=1700)


# ----------------------------------------------------------------------- voice story

_VOICE_DEMO = (
    "The last table at the corner was always the one my mother chose. It faced the window, "
    "where the afternoon light reached the chipped blue cup she carried from home. "
    "She never said the cup was special; she just kept bringing it.\n\n"
    "After she was gone, I found the cup wrapped in a dish towel in the back of my cabinet. "
    "I started taking it to the same table on Sundays. The coffee tastes no different there, "
    "but I have learned that a small ritual can hold a whole conversation."
)


def _voice_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:  # no Streamlit secrets file is a normal local setup
        return None


def voice_story_tab() -> None:
    """Let a writer make a keepsake from their words; models only provide optional signals."""
    from forge.experience.voice_story import (
        capsule_from_answers,
        parsed_quote_candidates,
        quote_candidates,
        render_capsule_html,
    )

    st.subheader("A voice, kept")
    st.markdown(
        "**A detector can notice patterns. It cannot know what a sentence means to you.** "
        "Choose a line, tell its story in your own words, and make something worth keeping."
    )
    st.caption(
        "Your writing stays in this app session unless you explicitly send it to Reducto or "
        "Pangram below. Forge does not save the document or the keepsake. The built-in "
        "sample is fictional."
    )

    source_mode = st.radio(
        "Choose your starting point",
        ("Paste writing", "Open a document"),
        horizontal=True,
        key="voice_source_mode",
    )
    parsed = st.session_state.get("voice_story_parse")
    source_name = "Pasted text"
    if source_mode == "Open a document":
        upload = st.file_uploader(
            "Choose a PDF, DOCX, or plain-text file",
            type=["pdf", "docx", "txt"],
            key="voice_story_upload",
        )
        if upload is not None and upload.name.lower().endswith(".txt"):
            try:
                st.session_state["voice_story_text"] = upload.getvalue().decode("utf-8-sig")
                st.session_state["voice_story_parse"] = None
                source_name = upload.name
            except UnicodeDecodeError:
                st.error("This text file is not UTF-8 encoded. Save it as UTF-8 and try again.")
        elif upload is not None:
            source_name = upload.name
            reducto_key = _voice_secret("REDUCTO_API_KEY")
            consent = st.checkbox(
                "I consent to send this document to Reducto for text extraction.",
                key="voice_reducto_consent",
            )
            if not reducto_key:
                st.info("Add REDUCTO_API_KEY in the environment or Streamlit secrets to enable parsing.")
            if st.button("Extract text with Reducto", disabled=not (consent and reducto_key)):
                from forge.experience.providers import ProviderError, reducto_parse

                with st.spinner("Parsing the document with Reducto…"):
                    try:
                        result = reducto_parse(upload.getvalue(), upload.name, api_key=reducto_key)
                    except (ProviderError, ValueError) as error:
                        st.error(str(error))
                    else:
                        st.session_state["voice_story_parse"] = result
                        st.session_state["voice_story_text"] = result["text"]
                        st.session_state["voice_story_source_name"] = upload.name
                        st.rerun()
        parsed = st.session_state.get("voice_story_parse")
        source_name = st.session_state.get("voice_story_source_name", source_name)

    if "voice_story_text" not in st.session_state:
        st.session_state["voice_story_text"] = _VOICE_DEMO
    text = st.text_area(
        "Your writing",
        key="voice_story_text",
        height=210,
        help="The fictional sample is ready to explore. Replace it with your own writing when you are ready.",
    )
    if source_mode == "Paste writing":
        source_name = "Pasted writing"

    if not text.strip():
        st.info("Add a passage to begin.")
        return

    current_digest = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    quote_options = (
        parsed_quote_candidates(parsed)
        if parsed and parsed.get("text") == text
        else quote_candidates(text)
    )
    st.markdown("#### 1 · Choose the line you want to keep")
    if quote_options:
        selected = st.selectbox(
            "Exact excerpts from your text",
            quote_options,
            format_func=lambda item: (
                f"{item.text[:115]}{'…' if len(item.text) > 115 else ''}"
                + (f" · page {item.page}" if item.page is not None else "")
            ),
            key="voice_quote_choice",
        )
        excerpt = selected.text
        source_page = selected.page
        if st.checkbox("I want to choose a different line myself", key="voice_manual_excerpt"):
            excerpt = st.text_area("Your chosen line", value=excerpt, max_chars=240)
            source_page = None
    else:
        excerpt = st.text_area("Write or paste a line to keep", max_chars=240)
        source_page = None

    st.markdown("#### 2 · Tell the part only you know")
    author = st.text_input("Name or initials (optional)", max_chars=80)
    title = st.text_input("Give this memory a title", max_chars=100, placeholder="The blue cup")
    detail = st.text_area(
        "What small detail do you remember most clearly?",
        placeholder="A sound, a gesture, a place, or something you noticed…",
        max_chars=1000,
        key="voice_detail",
    )
    intention = st.text_area(
        "What did you hope someone would feel or understand?",
        placeholder="Your intention, in your words…",
        max_chars=1000,
        key="voice_intention",
    )
    keepsake = st.text_area(
        "What would you want a future reader to remember?",
        placeholder="A message you want to leave with them…",
        max_chars=1000,
        key="voice_keepsake",
    )

    with st.expander("Optional signals · never a verdict about the writer"):
        st.write(
            "These checks can help inspect the text. They cannot verify who wrote it or why. "
            "Each provider sees the text only if you choose to send it."
        )
        pangram_key = _voice_secret("PANGRAM_API_KEY")
        pangram_consent = st.checkbox(
            "I consent to send this text to Pangram for segment-level AI-assistance signals.",
            key="voice_pangram_consent",
        )
        if not pangram_key:
            st.caption("Pangram is not connected. Set PANGRAM_API_KEY to enable the live comparison.")
        if st.button("Check with Pangram", disabled=not (pangram_consent and pangram_key)):
            from forge.experience.providers import ProviderError, pangram_analyze

            with st.spinner("Requesting Pangram signals…"):
                try:
                    result = pangram_analyze(text, api_key=pangram_key)
                except (ProviderError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.session_state["voice_pangram_result"] = {
                        "digest": current_digest,
                        "result": result,
                    }

        local_check = st.checkbox(
            "Also run Forge's matched-mirror text arm locally.",
            key="voice_local_consent",
            help="The text is scored in this app process. The first run may fetch the model weights.",
        )
        if st.button("Run Forge locally", disabled=not local_check):
            with st.spinner("Loading the Forge mirror arm and scoring this passage…"):
                try:
                    failure = ensure_weights("mirror")
                    if failure:
                        raise RuntimeError(f"could not load the mirror checkpoint: {failure}")
                    from forge.inference.scorer import load_arm

                    if st.session_state.get("_forge_loaded_arm") != "mirror":
                        load_arm.cache_clear()
                        st.session_state["_forge_loaded_arm"] = "mirror"
                    arm_result = load_arm("mirror").score(text)
                    st.session_state["voice_forge_result"] = {
                        "digest": current_digest,
                        "probability": arm_result.mean,
                        "windows": arm_result.window_probabilities,
                    }
                except Exception as error:
                    st.error(f"Forge could not score this passage: {type(error).__name__}: {error}")

        pangram = st.session_state.get("voice_pangram_result")
        forge = st.session_state.get("voice_forge_result")
        valid_pangram = pangram and pangram.get("digest") == current_digest
        valid_forge = forge and forge.get("digest") == current_digest
        if valid_pangram or valid_forge:
            st.markdown("##### Signals for this passage")
            left, right = st.columns(2)
            if valid_pangram:
                result = pangram["result"]
                with left:
                    st.metric("Pangram · " + str(result.get("model")), str(result.get("prediction") or "Analysis complete"))
                    st.caption(
                        "AI-like " + str(result.get("fraction_ai"))
                        + " · assisted " + str(result.get("fraction_ai_assisted"))
                        + " · human-like " + str(result.get("fraction_human"))
                    )
                    for window in result.get("windows", []):
                        if window.get("text"):
                            st.caption(f"{window.get('label')}: {window['text'][:180]}")
            if valid_forge:
                with right:
                    st.metric("Forge · matched-mirror arm", f"{forge['probability']:.1%} AI-pattern score")
                    st.caption("Research model signal, not an authorship finding or calibrated probability.")
                    st.caption(f"Scored {len(forge['windows'])} text windows.")
            st.info("Differences are a reason to review context. They are not evidence of misconduct or proof of human authorship.")

    st.markdown("#### 3 · Make the keepsake")
    if st.button("Create my voice story", type="primary"):
        try:
            from forge.experience.voice_story import capsule_from_answers

            capsule = capsule_from_answers(
                title=title,
                author=author,
                excerpt=excerpt,
                source_page=source_page,
                detail=detail,
                intention=intention,
                keepsake=keepsake,
                source_name=source_name,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["voice_capsule"] = capsule.as_dict()
            st.session_state["voice_capsule_text_digest"] = current_digest

    capsule_data = st.session_state.get("voice_capsule")
    if capsule_data:
        from forge.experience.voice_story import VoiceCapsule, render_capsule_html

        capsule = VoiceCapsule(**capsule_data)
        if st.session_state.get("voice_capsule_text_digest") != current_digest:
            # A keepsake belongs to the exact source passage used to create it.
            st.session_state.pop("voice_capsule", None)
            st.info("Your source text changed; create a fresh keepsake for the new passage.")
        else:
            card = render_capsule_html(capsule)
            st.markdown("#### Your story, in your own voice")
            components.html(card, height=1050, scrolling=True)
            st.download_button(
                "Download the narrated keepsake",
                data=card,
                file_name="voice-story.html",
                mime="text/html",
            )
            st.download_button(
                "Download the story data",
                data=json.dumps(capsule.as_dict(), ensure_ascii=False, indent=2),
                file_name="voice-story.json",
                mime="application/json",
            )
            st.caption("The HTML works offline. Its Listen button uses your browser's built-in speech voice.")
    if st.button("Start a new story"):
        for key in ("voice_capsule", "voice_capsule_text_digest", "voice_pangram_result", "voice_forge_result"):
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------------- page

st.title("FORGE Detect")

# THREE TASKS, TWO DETECTORS. The Engineering tab rendered thirteen artifact-backed panels,
# which is the
# right content in the wrong place: a visitor here wants a verdict on their text or their
# image, and the measurements belong in docs/evidence.md and the generated evidence page
# where they can be read properly. Removing it also removes every artifact read from the
# deployed page's startup path.
text_pane, image_pane, story_pane = st.tabs(["Text", "Image", "Voice story"])
with text_pane:
    text_tab()
with image_pane:
    image_tab()
with story_pane:
    voice_story_tab()

st.divider()
# THE FOOTER SAID SOMETHING FALSE. It explained that one arm was held in memory at a time,
# which was true of an earlier draft of this page and stopped being true when both arms were
# restored. A page that describes its own behaviour incorrectly is the same defect as a
# verdict that does not match its evidence, and this project has shipped that often enough
# to know it. The byline is what belongs here; the memory constraint is an implementation
# note and lives in the module docstring.
st.caption(
    "Built by Akila Lourdes Miriyala Francis · github.com/AKilalours/Forge"
)
