"""
Confirms the startup dependency check passes and that it reports a
useful message when a package really is missing.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT)
)


from core.dependencies import (
    missing_packages,
    ensure_dependencies,
    MissingDependencyError
)


def test_no_required_package_is_missing():
    """
    The environment must satisfy every entry in requirements.txt.
    """
    assert missing_packages() == []


def test_ensure_dependencies_is_silent_when_satisfied():
    ensure_dependencies()


def test_missing_dependency_names_the_interpreter():
    """
    The message must say what is missing, which interpreter is
    running, and the exact command that fixes it.
    """
    probe = (
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "import core.dependencies as d\n"
        "d.REQUIRED_PACKAGES = (\n"
        "    ('definitely_not_installed_xyz', 'nope'),\n"
        ")\n"
        "try:\n"
        "    d.ensure_dependencies()\n"
        "    print('NO ERROR RAISED')\n"
        "except d.MissingDependencyError as e:\n"
        "    print('MESSAGE:')\n"
        "    print(str(e))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(ROOT),
        capture_output=True,
        text=True
    )

    message = result.stdout

    assert "NO ERROR RAISED" not in message
    assert "nope" in message
    assert sys.executable in message
    assert "pip install" in message


def test_edge_tts_is_installed():
    """
    Regression guard.

    edge-tts drives the narration, and the SnapGenAI requirements.txt
    did not list it, so an environment built from that file could start
    the server and then fail every job with a bare ModuleNotFoundError.
    """
    import edge_tts

    assert edge_tts is not None