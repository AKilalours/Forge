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
    arm = st.radio(
        "Arm", ("mirror", "baseline"), horizontal=True,
        help="One at a time. Switching evicts the other rather than accumulating, because "
             "two arms do not fit in this host's memory. The two-arm comparison is in "
             "docs/evaluation.md.",
    )
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
            payload = analyse(data, filename=upload.name, with_stability=False,
                              with_diagnostics=False)
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
    show(image_result(payload, compact=True), height=560)


# ---------------------------------------------------------------------------------- page

st.title("FORGE Detect")

# TWO TABS. The Engineering tab rendered thirteen artifact-backed panels, which is the
# right content in the wrong place: a visitor here wants a verdict on their text or their
# image, and the measurements belong in docs/evidence.md and the generated evidence page
# where they can be read properly. Removing it also removes every artifact read from the
# deployed page's startup path.
text_pane, image_pane = st.tabs(["Text", "Image"])
with text_pane:
    text_tab()
with image_pane:
    image_tab()

st.divider()
# THE FOOTER SAID SOMETHING FALSE. It explained that one arm was held in memory at a time,
# which was true of an earlier draft of this page and stopped being true when both arms were
# restored. A page that describes its own behaviour incorrectly is the same defect as a
# verdict that does not match its evidence, and this project has shipped that often enough
# to know it. The byline is what belongs here; the memory constraint is an implementation
# note and lives in the module docstring.
st.caption(
    "Built by Akila Lourdes Miriyala Francis · github.com/AKilalours/Panagram_Forge"
)
