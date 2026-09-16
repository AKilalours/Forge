"""THE REGRESSION. Two hand-maintained dependency lists disagreed, and the wrong one won.

`space/requirements.txt` listed `python-multipart`. The `serve` extra in `pyproject.toml`
did not. FastAPI needs it to build the route for `/v1/image/analyze`, and it raises at
IMPORT time rather than at request time, so the failure is not a degraded endpoint: the
process does not start. That meant `pip install -e ".[serve]"` produced an installation
that could not boot the app, while the deployment target booted fine. The package was
wrong and the thing furthest from the tests was right.

The fix is not "add the package". It is to make the two lists unable to disagree: anything
the deployed Space installs must also be declared in the package, so a dependency added for
deployment cannot skip the package that developers and CI install from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:  # 3.11+ ships tomllib; the package supports 3.10, where tomli is the equivalent.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter, not the code
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli on 3.10")

ROOT = Path(__file__).resolve().parents[2]


def _normalise(name: str) -> str:
    """PEP 503 names: case-insensitive, and -, _ and . are all the same character."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(text: str) -> set[str]:
    names = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0]
        if name:
            names.add(_normalise(name))
    return names


def _declared_names() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    lists = [project.get("dependencies", [])]
    lists += list(project.get("optional-dependencies", {}).values())
    names = set()
    for group in lists:
        for spec in group:
            names.add(_normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]))
    return names


def test_everything_the_space_installs_is_declared_in_the_package():
    space = _requirement_names((ROOT / "space" / "requirements.txt").read_text(encoding="utf-8"))
    missing = sorted(space - _declared_names())
    assert not missing, (
        "space/requirements.txt installs packages pyproject.toml never declares: "
        f"{missing}. A developer running `pip install -e '.[serve,image,train]'` gets an "
        "installation the deployed app does not match."
    )


def test_multipart_is_a_declared_serve_dependency():
    """Named explicitly, because its absence is a boot failure and not a missing feature."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    serve = {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"]["optional-dependencies"]["serve"]
    }
    assert "python-multipart" in serve, (
        "fastapi raises at import time on a multipart route without it; the app will not start"
    )
    assert "pillow" in serve, "every image upload is decoded with Pillow on the serve path"


# --------------------------------------------------------------- the serve extra must serve

# The modules a running server reaches: the HTTP layer, and the two detectors plus the
# attribution pass it calls into. Add a module here when the server starts importing it.
SERVING_MODULES = (
    "api/forge_app.py",
    "api/results.py",
    "src/forge/inference/scorer.py",
    "src/forge/inference/decision.py",
    "src/forge/image/detector.py",
    "src/forge/image/attribution.py",
    "src/forge/image/report.py",
    "src/forge/image/evidence.py",
    "src/forge/image/maps.py",
)

# Import name to distribution name, for the few where they differ.
DISTRIBUTION = {"PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn"}


def _third_party_imports(path: Path) -> set[str]:
    """Every non-stdlib top-level import, including imports inside functions.

    Function-level imports matter more than usual here: the server defers torch and
    transformers into request handlers to keep startup fast, so a naive scan of the file
    header would have found nothing and concluded the serve extra was complete.
    """
    import ast
    import sys

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return {
        name for name in found
        if name not in sys.stdlib_module_names and name not in {"forge", "api", "__future__"}
    }


def test_the_serve_extra_can_actually_serve():
    """THE REGRESSION. `pip install -e '.[serve,image,dev]'` produced a server with no models.

    torch and transformers were declared only under `train`, because they were thought of
    as training dependencies. They are not: the text arms, the visual detector and the
    occlusion attribution pass all run on them at request time. The app booted normally and
    every detector reported "not loaded", so the interface looked like a working build with
    a configuration problem, which is the most expensive way for this to fail.

    Deriving the list from the code rather than restating it means the next module the
    server starts importing is checked automatically.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    serve = {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"]["optional-dependencies"]["serve"]
    }
    serve |= {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"].get("dependencies", [])
    }

    needed: dict[str, str] = {}
    for relative in SERVING_MODULES:
        path = ROOT / relative
        if not path.exists():
            continue
        for name in _third_party_imports(path):
            needed.setdefault(_normalise(DISTRIBUTION.get(name, name)), relative)

    missing = sorted(f"{dist} (imported by {where})" for dist, where in needed.items()
                     if dist not in serve)
    assert not missing, (
        "the serve extra does not declare what the serving path imports: "
        + "; ".join(missing)
        + ". Installing '.[serve]' would give a server that boots and cannot load a model."
    )


def test_the_tokenizer_backend_is_declared_wherever_the_text_arm_is_served():
    """THE REGRESSION. torch and transformers were present and the text arm still failed.

    The DeBERTa-v3 backbone tokenizer is a SentencePiece model. Without `sentencepiece`,
    transformers falls back to a TikToken converter, that import fails too, and
    AutoTokenizer.from_pretrained raises. The arm reports unavailable on a machine that has
    the entire model stack installed, which reads as a broken checkpoint rather than a
    missing package.

    It is not discoverable by scanning imports: no file in this project imports
    sentencepiece. It is a runtime backend transformers reaches for, so it has to be
    asserted by name in both places the app is installed from.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    serve = {
        _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0])
        for spec in data["project"]["optional-dependencies"]["serve"]
    }
    assert "sentencepiece" in serve, "the text arm cannot build its tokenizer without it"

    space = _requirement_names((ROOT / "space" / "requirements.txt").read_text(encoding="utf-8"))
    assert "sentencepiece" in space, "the deployed Space would serve a dead text tab"


def test_transformers_is_pinned_below_5_in_both_installs():
    """The arms' thresholds were fitted under transformers 4.x tokenisation.

    Version 5 rewrote tokenizer construction. A threshold is only meaningful against the
    tokenisation it was measured with, so an unpinned upper bound silently swaps the
    tokeniser under a fixed decision boundary. Unpin this only together with a re-run of
    the evaluation and a new record under reports/experiments.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = [
        spec for group in
        [data["project"].get("dependencies", [])]
        + list(data["project"].get("optional-dependencies", {}).values())
        for spec in group
        if _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]) == "transformers"
    ]
    assert specs, "transformers is not declared anywhere"
    for spec in specs:
        assert "<5" in spec, f"unbounded transformers requirement: {spec!r}"

    space_text = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    line = [ln for ln in space_text.splitlines() if ln.strip().startswith("transformers")]
    assert line and "<5" in line[0], f"the Space would install transformers 5: {line}"


def test_huggingface_hub_is_pinned_below_1_in_both_installs():
    """THE REGRESSION. `pip install -U huggingface_hub` installed 1.30 and broke the venv.

    transformers 4.x requires huggingface-hub<1.0. Both requirements were written with a
    lower bound and no ceiling, so the resolver was free to install a major version the
    pinned transformers cannot use. pip printed the conflict and installed it anyway.

    Locally that is a visible error. In the Space build it would be a container that starts,
    reports the text arms unavailable, and gives no obvious reason, on a machine nobody is
    watching.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = [
        spec for group in
        [data["project"].get("dependencies", [])]
        + list(data["project"].get("optional-dependencies", {}).values())
        for spec in group
        if _normalise(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]) == "huggingface-hub"
    ]
    assert specs, "huggingface-hub is not declared anywhere"
    for spec in specs:
        assert "<1" in spec, f"unbounded huggingface-hub requirement: {spec!r}"

    space_text = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    line = [ln for ln in space_text.splitlines() if ln.strip().startswith("huggingface_hub")]
    assert line and "<1.0" in line[0], f"the Space would install huggingface_hub 1.x: {line}"


# ------------------------------------------------------- the two deployment requirement files

def test_the_streamlit_requirements_agree_with_the_space_requirements():
    """Two deployment files, one set of pins. They must not drift.

    Streamlit Community Cloud reads requirements.txt at the repository root and nothing
    else, so the pins that keep the text arm alive (transformers below 5, sentencepiece,
    huggingface_hub below 1) have to be repeated there. A repeated pin is a pin that rots:
    this project has already shipped python-multipart in one file and not the other, and a
    dead text tab is exactly what that produces.

    Shared packages must carry identical specifiers. Either file may add packages the other
    does not need, which is how streamlit and onnxruntime differ legitimately.
    """
    def specs(text: str) -> dict[str, str]:
        out = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0]
            out[_normalise(name)] = line[len(name):].strip()
        return out

    root = specs((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    space = specs((ROOT / "space" / "requirements.txt").read_text(encoding="utf-8"))

    disagreements = {
        name: (root[name], space[name])
        for name in set(root) & set(space)
        if root[name] != space[name]
    }
    assert not disagreements, (
        "requirements.txt and space/requirements.txt disagree on shared packages: "
        f"{disagreements}"
    )
    for required in ("torch", "transformers", "sentencepiece", "huggingface-hub", "pillow"):
        assert required in root, f"{required} missing from the Streamlit requirements"
    assert "streamlit" in root, "the Streamlit deployment cannot run without streamlit"


def test_the_streamlit_app_declares_the_same_ceilings():
    """The pins that matter are the upper bounds, so assert them by name in the root file."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = {ln.split("#")[0].strip() for ln in text.splitlines()}
    assert any(ln.startswith("transformers") and "<5" in ln for ln in lines)
    assert any(ln.startswith("huggingface_hub") and "<1.0" in ln for ln in lines)


# ---------------------------------------------------------------------------
# THE SECOND REGRESSION, same shape as the first, found by the README badge check.
#
# tests/unit/test_forge_app.py guards itself with four module-level importorskip calls:
# PIL, numpy, fastapi and httpx. httpx was declared in no extra, in no requirements file,
# nowhere. It was installed on the author's machine as somebody else's transitive
# dependency, so the module ran there and skipped entirely in CI.
#
# That is worse than a missing dependency, because a module-level importorskip produces a
# GREEN run. The tests do not fail, they cease to exist, and the job reports success. This
# repo had already been bitten by exactly that once: an earlier fix added the `serve` extra
# so this same module would run in CI, which satisfied the fastapi guard and stopped one
# guard short.
#
# The rule below is the general form: if a test module refuses to run without a package,
# that package must be declared, so installing the project can actually satisfy it. A test
# guarded on a dependency nobody declares is a test nobody runs.
# ---------------------------------------------------------------------------

# importorskip takes IMPORT names; pyproject declares DISTRIBUTION names. Only the ones
# that genuinely differ belong here, and the mapping is explicit so a wrong guess fails
# loudly rather than quietly excusing an undeclared package.
# Deliberately minimal: only names this repo actually gates a module on today. A
# speculative entry for a package nobody declares would be a pre-authorised excuse, which
# is what test_the_mapping_itself_is_not_stale below exists to prevent. Add a line when a
# guard needs one, not before.
IMPORT_TO_DISTRIBUTION = {
    "PIL": "pillow",
}

# Modules the standard library provides on every supported version. tomllib is NOT here:
# it is stdlib only from 3.11, and on 3.10 it needs the tomli distribution, which is
# exactly the second gap this rule caught.
STDLIB_SAFE = {"json", "sqlite3", "tomllib"}


def _module_level_importorskips() -> dict[str, set[str]]:
    """{test file: {import names it refuses to run without}}.

    Only module level. A call inside a function skips one test and is a deliberate,
    visible choice; a call at import time deletes the whole file from the run.
    """
    found: dict[str, set[str]] = {}
    pattern = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?_?pytest\.importorskip\(\s*['\"]([^'\"]+)['\"]")
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        names = {
            m.group(1).split(".")[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if (m := pattern.match(line))
        }
        if names:
            found[str(path.relative_to(ROOT))] = names
    return found


def test_every_module_level_importorskip_names_a_declared_package() -> None:
    """A whole test file may only be gated on something the project can install.

    This is the check that would have caught httpx. It fails on the name of the package
    and the file it silently disabled, because "some tests did not run" is not a message
    anyone acts on.
    """
    declared = _declared_names()
    undeclared: list[str] = []
    for test_file, imports in _module_level_importorskips().items():
        for import_name in sorted(imports):
            if import_name in STDLIB_SAFE:
                continue
            dist = _normalise(IMPORT_TO_DISTRIBUTION.get(import_name, import_name))
            if dist not in declared:
                undeclared.append(
                    f"{test_file} skips itself entirely without {import_name!r} "
                    f"(distribution {dist!r}), which pyproject.toml does not declare. "
                    "Installing this project can never satisfy that guard, so the file "
                    "runs only where the package arrived by accident."
                )
    assert not undeclared, "test modules gated on undeclared packages:\n  " + "\n  ".join(undeclared)


def test_the_mapping_itself_is_not_stale() -> None:
    """Every entry in IMPORT_TO_DISTRIBUTION must point at something actually declared.

    Otherwise the mapping becomes a way to excuse an undeclared package: add a line here
    and the check above stops complaining without anything being fixed.
    """
    declared = _declared_names()
    stale = [
        f"{imp} -> {dist}" for imp, dist in IMPORT_TO_DISTRIBUTION.items()
        if _normalise(dist) not in declared
    ]
    assert not stale, (
        "IMPORT_TO_DISTRIBUTION maps to packages pyproject.toml does not declare, which "
        f"would let an undeclared dependency pass as mapped: {stale}"
    )
