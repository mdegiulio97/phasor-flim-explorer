"""Smoke test dell'app Streamlit: lo script gira end-to-end senza eccezioni."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_runs_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    # L'app espone la metrica del MAE e almeno un titolo.
    assert any("Phasor" in t.value for t in at.title)
    assert len(at.metric) >= 1
