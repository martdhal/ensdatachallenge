from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]

def test_demo_and_filters():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "160"
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert any("No patients" in x.value for x in app.info)

def test_model_button():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any(x.label == "Held-out Harrell C-index" for x in app.metric)

def test_upload_empty_state():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Upload CSV files").run()
    assert not app.exception
    assert any("Upload a clinical CSV" in x.value for x in app.info)
