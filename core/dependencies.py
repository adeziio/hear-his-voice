"""
Startup dependency check.

A missing package otherwise surfaces deep inside a job as a bare
`ModuleNotFoundError`, which does not say which interpreter is running
or how to fix it. This reports every missing package at once, together
with the interpreter in use, so the cause is obvious immediately.
"""

import importlib.util
import sys


class MissingDependencyError(
    RuntimeError
):

    pass


# Every third-party module the pipeline needs before it can do any
# work. Kept in step with requirements.txt.
REQUIRED_PACKAGES = (
    ("requests", "requests"),
    ("selenium", "selenium"),
    ("edge_tts", "edge-tts"),
    ("moviepy", "moviepy"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
)


def missing_packages():
    """
    Returns [(import_name, pip_name)] for anything not importable.
    """
    missing = []

    for import_name, pip_name in REQUIRED_PACKAGES:

        try:

            found = importlib.util.find_spec(
                import_name
            )

        except (ImportError, ValueError):

            found = None

        if found is None:

            missing.append(
                (import_name, pip_name)
            )

    return missing


def ensure_dependencies():
    """
    Raises a single, actionable error listing every missing package.
    """
    missing = missing_packages()

    if not missing:

        return

    names = ", ".join(
        pip_name
        for _, pip_name in missing
    )

    raise MissingDependencyError(
        f"Missing required package(s): {names}.\n"
        f"These are running with: {sys.executable}\n"
        "Install them into that environment, for example:\n"
        f'    "{sys.executable}" -m pip install -r requirements.txt'
    )