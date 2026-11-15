from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_without_exception():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "📚 Adaptive Study Planner"


def test_sample_adaptive_update_and_baseline_flow():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.run()
    app.button(key="load_sample").click().run()
    assert not app.exception
    assert len(app.session_state["topics"]) == 5

    app.button(key="generate_adaptive").click().run()
    assert not app.exception
    assert app.session_state["adaptive_result"] is not None
    assert app.session_state["adaptive_result"].sessions

    app.selectbox(key="update_topic_select").set_value("CS101 / Recursion").run()
    score_widget = next(
        widget
        for widget in app.number_input
        if widget.key and widget.key.startswith("new_score_")
    )
    score_widget.set_value(80).run()
    app.button(key="update_score").click().run()
    assert not app.exception
    assert app.session_state["before_snapshot"] is not None
    assert app.session_state["after_snapshot"] is not None

    app.button(key="generate_baseline").click().run()
    assert not app.exception
    assert app.session_state["baseline_result"] is not None
