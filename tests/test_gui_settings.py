"""GUI-only settings tests: no real credentials, LLM calls or literature access."""

import io
import json
import os
import urllib.error

import pytest
from streamlit.testing.v1 import AppTest

from src import dashboard, db
from src.gui.settings import load_model_ids, validate_endpoint
from src.llm import LLMConfiguration
from src.schemas import ResearchTask, ResearchUnit


def widget(app, kind, key):
    return next(item for item in getattr(app, kind) if item.key == key)


def button(app, label):
    return next(item for item in app.button if item.label == label)


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "gui-settings.db"
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "placeholder")
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("TEST_DATABASE_URL", db.resolve_database_url())
    monkeypatch.setenv("DATABASE_URL", str(path))
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Test objective"))
        db.insert_research_unit(connection, ResearchUnit(
            task_id=task_id, kind="analysis", title="A long title " * 20,
            purpose="Inspect this unit", status="blocked",
        ))
    return AppTest.from_file(str(db.PROJECT_ROOT / "src/gui/app.py")).run(timeout=10)


def test_session_key_survives_navigation_and_is_used_only_for_runs(app, monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["client"])
        return {"status": "completed", "message": "Done", "research_unit_ids": []}
    monkeypatch.setattr(dashboard, "run_research_steps", run)
    app.radio[0].set_value("Settings").run(timeout=10)
    widget(app, "selectbox", "settings_api_mode").set_value("OpenAI-compatible API")
    widget(app, "text_input", "settings_api_url").set_value("https://example.org/v1")
    widget(app, "text_input", "settings_api_key").set_value("fake-session-secret")
    button(app, "Apply connection").click().run(timeout=10)
    assert not app.exception
    assert widget(app, "text_input", "settings_api_key").value == ""
    assert "LLM_API_KEY" not in os.environ
    assert not calls
    app.radio[0].set_value("Research Tasks").run(timeout=10)
    assert not app.exception
    widget(app, "button", "continue_research").click().run(timeout=10)
    assert calls[0].configuration.api_key == "fake-session-secret"
    assert calls[0].configuration.base_url == "https://example.org/v1"
    app.radio[0].set_value("System Status").run(timeout=10)
    assert all("fake-session-secret" not in str(item.value) for item in app.json)
    app.radio[0].set_value("Settings").run(timeout=10)
    assert widget(app, "text_input", "settings_api_key").value == ""
    # A new endpoint must not silently inherit the previous endpoint's key.
    widget(app, "text_input", "settings_api_url").set_value("https://different.example/v1")
    button(app, "Apply connection").click().run(timeout=10)
    assert any("Enter a key" in item.value for item in app.error)
    assert app.session_state["api_configuration"].base_url == "https://example.org/v1"
    widget(app, "button", "reset_settings").click().run(timeout=10)
    assert not app.exception
    assert "api_configuration" not in app.session_state


def test_graph_is_compact_and_table_is_secondary(app):
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.get("bidi_component")
    assert any("Colours show activity type" in caption.value for caption in app.caption)
    assert any(expander.label == "All units (1)" and not expander.proto.expanded
               for expander in app.expander)
    assert not any(item.key == "llm_model_proof_attempt" for item in app.text_input)


def test_graph_click_updates_only_inspector(app, monkeypatch):
    import streamlit as st

    with db.get_connection(os.environ["DATABASE_URL"]) as connection:
        unit_id = db.insert_research_unit(connection, ResearchUnit(
            task_id=1, parent_unit_id=1, kind="proof_attempt", title="Inspect a proof",
            purpose="Try induction", status="finished", outcome_note="Induction gap remains",
        ))
    clicked = [unit_id]

    def component(**kwargs):
        st.session_state[kwargs["key"]] = {"clicked": clicked[0]}
        kwargs["on_clicked_change"]()

    monkeypatch.setattr(st.components.v2, "component", lambda *args, **kwargs: component)
    app.run(timeout=10)
    assert not app.exception
    assert widget(app, "selectbox", "inspect_unit_1").value == unit_id
    assert any("proof attempt" in item.value for item in app.caption)
    assert any("Induction gap remains" in item.value for item in app.text)
    assert widget(app, "selectbox", "unit_work_selector").value is None
    clicked[0] = 999999
    app.run(timeout=10)
    assert not app.exception
    assert widget(app, "selectbox", "inspect_unit_1").value == unit_id


def test_model_catalog_request_is_explicit_and_bounded(monkeypatch):
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(json.dumps({"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}]}).encode())
    monkeypatch.setattr("src.gui.settings.urllib.request.build_opener", lambda *_: Opener())
    configuration = LLMConfiguration(provider="openai-compatible", api_key="fake", base_url="https://example.org/v1")
    assert load_model_ids(configuration) == ["a", "b"]
    assert requests[0][0].full_url == "https://example.org/v1/models"
    assert requests[0][0].get_method() == "GET"
    assert requests[0][1] == 15


def test_model_catalog_errors_do_not_echo_credentials(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 401, "fake-secret", {}, None)
    monkeypatch.setattr("src.gui.settings.urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(ValueError, match="HTTP 401") as error:
        load_model_ids(LLMConfiguration(provider="openai", api_key="fake-secret"))
    assert "fake-secret" not in str(error.value)


@pytest.mark.parametrize("url", ["http://example.org/v1", "https://user:secret@example.org", "https://example.org?key=secret"])
def test_settings_reject_unsafe_endpoint_forms(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)
