# UI

> **Legacy mockup, not served by the current application.** The live interface is
> [`streamlit_app.py`](../streamlit_app.py); the FastAPI page is implemented in
> [`api/forge_app.py`](../api/forge_app.py). This mockup calls the old `/v1/model` and
> `/v1/detect` API and is retained as a design-history artifact.

Phase 8. Deliberately minimal: a text box, an Analyze button, and a result panel
showing prediction, AI probability, token-level highlighting, model version, and
whether the score is calibrated.

Two things the UI must always show, because a detector that hides them is misleading:
the model version, and the operating threshold's FPR budget. A user seeing "AI, 96
percent confidence" deserves to know what that 96 percent was calibrated against.
