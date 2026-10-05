"""
Shared test fixtures.

The important one here is isolation of the Scripture rotation state.

Selecting an episode writes the reading position to
state/scripture_rotation.json, which is the channel's REAL progress
through the Gospels - committed to the repository, and the thing that
keeps the reading from repeating. A test that calls select() without
redirecting that path silently rewinds the real channel to wherever the
test left it.

This is not hypothetical: two tests once consumed real rotation state, and
the channel was left part-way through Mark rather than at the beginning of
Matthew. So the guard below applies to EVERY test, and additionally fails
the test outright if the real file was written during it.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_scripture_rotation(monkeypatch, tmp_path):
    """
    Redirects Scripture rotation state to a temporary file for every test,
    and fails the test if the real state file was touched anyway.
    """

    import scripture.selector as selector_module

    from pathlib import Path

    real_path = Path(selector_module.STATE_PATH)

    real_before = (
        real_path.read_bytes()
        if real_path.is_file()
        else None
    )

    redirected = tmp_path / "scripture_rotation.json"

    monkeypatch.setattr(
        selector_module,
        "STATE_PATH",
        str(redirected),
    )

    yield str(redirected)

    real_after = (
        real_path.read_bytes()
        if real_path.is_file()
        else None
    )

    assert real_after == real_before, (
        f"a test modified the real rotation state at {real_path}. "
        "Tests must use the redirected path, never the live one."
    )