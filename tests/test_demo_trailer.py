"""Statistics displayed by the trailer must stay within a percentage's range."""
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(("lines", "expected"), [
    (["#acc tokens = 57, #gen tokens = 100"], 57),
    (["#acc tokens = 100, #gen tokens = 80"], 100),
    (["#acc tokens = 0, #gen tokens = 0"], None),
    (["no statistics recorded"], None),
])
def test_draft_rate(monkeypatch, lines, expected):
    demo = Path(__file__).resolve().parents[1] / "scripts" / "demo"
    monkeypatch.syspath_prepend(str(demo))
    trailer = runpy.run_path(str(demo / "trailer.py"))
    assert trailer["draft_rate"](lines) == expected
