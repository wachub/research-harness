"""GUI settings tests: temporary credentials only, with no LLM calls."""

import os
import stat

import pytest
from dotenv import dotenv_values
from streamlit.testing.v1 import AppTest

from src import dashboard, db
from src.gui import settings as gui_settings
from src.gui.settings import (configure_provider, load_gui_configuration,
                              provider_label, save_gui_configuration, validate_endpoint)
from src.llm import LLMConfiguration
from src.schemas import ResearchTask, ResearchUnit


def widget(app, kind, key):
    return next(item for item in getattr(app, kind) if item.key == key)


def button(app, label):
    return next(item for item in app.button if item.label == label)


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "gui-settings.db"
    monkeypatch.setattr(gui_settings, "ENV_PATH", tmp_path / ".env")
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


def test_saved_key_survives_navigation_and_new_session_and_is_used_only_for_runs(app, monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs["client"])
        return {"status": "completed", "message": "Done", "research_unit_ids": []}
    monkeypatch.setattr(dashboard, "run_research_steps", run)
    app.radio[0].set_value("Settings").run(timeout=10)
    widget(app, "text_input", "settings_provider").set_value("OpenAI")
    widget(app, "text_input", "settings_api_key").set_value("fake-session-secret")
    button(app, "Save API connection").click().run(timeout=10)
    assert not app.exception
    assert widget(app, "text_input", "settings_api_key").value == ""
    assert "LLM_API_KEY" not in os.environ
    saved = load_gui_configuration()
    assert saved.api_key == "fake-session-secret"
    assert saved.model == "gpt-4.1-mini"
    assert stat.S_IMODE(gui_settings.ENV_PATH.stat().st_mode) == 0o600
    assert not calls
    app.radio[0].set_value("Research Tasks").run(timeout=10)
    assert not app.exception
    widget(app, "button", "continue_research").click().run(timeout=10)
    assert calls[0].configuration.api_key == "fake-session-secret"
    assert calls[0].configuration.base_url == "https://api.openai.com/v1"
    app.radio[0].set_value("System Status").run(timeout=10)
    assert all("fake-session-secret" not in str(item.value) for item in app.json)
    app.radio[0].set_value("Settings").run(timeout=10)
    assert widget(app, "text_input", "settings_api_key").value == ""
    fresh = AppTest.from_file(str(db.PROJECT_ROOT / "src/gui/app.py")).run(timeout=10)
    assert fresh.session_state["api_configuration"].api_key == "fake-session-secret"
    assert fresh.session_state["api_configuration"].model == "gpt-4.1-mini"
    # A new endpoint must not silently inherit the previous endpoint's key.
    widget(app, "text_input", "settings_provider").set_value("Gemini")
    button(app, "Save API connection").click().run(timeout=10)
    assert any("Enter an API key" in item.value for item in app.error)
    assert app.session_state["api_configuration"].base_url == "https://api.openai.com/v1"
    assert load_gui_configuration().api_key == "fake-session-secret"


def test_graph_is_compact_and_table_is_secondary(app):
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.get("bidi_component")
    assert any("Colours show activity type" in caption.value for caption in app.caption)
    assert any(item.key == "inspect_unit_1" for item in app.selectbox)
    assert not any(item.key == "llm_model_proof_attempt" for item in app.text_input)


def test_graph_click_updates_only_inspector(app, monkeypatch):
    import research_unit_graph
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

    monkeypatch.setattr(research_unit_graph, "research_unit_graph", component)
    app.run(timeout=10)
    assert not app.exception
    assert widget(app, "selectbox", "inspect_unit_1").value == unit_id
    assert any("proof attempt" in item.value for item in app.caption)
    assert any("Induction gap remains" in item.value for item in app.markdown)
    assert widget(app, "selectbox", "unit_work_selector").value is None
    clicked[0] = 999999
    app.run(timeout=10)
    assert not app.exception
    assert widget(app, "selectbox", "inspect_unit_1").value == unit_id


def test_settings_have_no_model_or_provider_dropdowns_or_network_calls(app, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Settings must not call the API")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    app.radio[0].set_value("Settings").run(timeout=10)
    assert not app.selectbox
    assert not any("Load models" in item.label for item in app.button)
    assert any(item.label == "Custom endpoint" and not item.proto.expanded for item in app.expander)
    assert {item.key for item in app.text_input} == {
        "settings_provider", "settings_model", "settings_api_key", "settings_api_url",
    }
    widget(app, "text_input", "settings_provider").set_value("Gemini")
    widget(app, "text_input", "settings_api_key").set_value("fake-secret")
    button(app, "Save API connection").click().run(timeout=10)
    assert not app.exception
    assert app.session_state["api_configuration"].model == "gemini-2.5-flash"
    assert dotenv_values(gui_settings.ENV_PATH)["LLM_MODEL"] == "gemini-2.5-flash"


def test_custom_endpoint_requires_its_own_key_and_model(app):
    app.radio[0].set_value("Settings").run(timeout=10)
    widget(app, "text_input", "settings_provider").set_value("OpenAI")
    widget(app, "text_input", "settings_api_key").set_value("old-secret")
    button(app, "Save API connection").click().run(timeout=10)
    widget(app, "text_input", "settings_provider").set_value("custom")
    widget(app, "text_input", "settings_api_url").set_value("https://example.org/v1")
    widget(app, "text_input", "settings_model").set_value("custom-model")
    button(app, "Save API connection").click().run(timeout=10)
    assert not app.exception
    assert any("Enter an API key" in item.value for item in app.error)
    assert load_gui_configuration().base_url == "https://api.openai.com/v1"
    widget(app, "text_input", "settings_api_key").set_value("new-secret")
    button(app, "Save API connection").click().run(timeout=10)
    assert not app.exception
    assert app.session_state["api_configuration"].model == "custom-model"
    assert app.session_state["api_configuration"].api_key == "new-secret"
    assert app.session_state["api_configuration"].base_url == "https://example.org/v1"
    assert app.session_state["model_routes"] == {}
    assert widget(app, "text_input", "settings_api_key").value == ""


def test_provider_settings_preserve_same_endpoint_model_and_reject_unknown_names():
    config = LLMConfiguration("openai", "chosen-model", "old-secret")
    assert configure_provider(" OpenAI ", "", config) == LLMConfiguration("openai-compatible", "chosen-model", "old-secret")
    assert configure_provider("offline", "", config).api_key == ""
    assert provider_label(config) == "openai"
    with pytest.raises(ValueError, match="Enter an API key"):
        configure_provider("Gemini", "", config)
    with pytest.raises(ValueError, match="Enter OpenAI"):
        configure_provider("unknown-provider", "new-secret", config)


@pytest.mark.parametrize("url", ["http://example.org/v1", "https://user:secret@example.org", "https://example.org?key=secret"])
def test_settings_reject_unsafe_endpoint_forms(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)


def test_save_preserves_unrelated_env_values_and_removes_old_llm_duplicates(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DATABASE_URL=postgresql:///research_harness\nOPENALEX_API_KEY=unrelated\n"
                    "LLM_MODEL=old\nLLM_MODEL=older\n# keep this comment\n", encoding="utf-8")
    configuration = LLMConfiguration("openai-compatible", "chosen-model", "new-secret",
                                     "https://example.org/v1")
    save_gui_configuration(configuration, path)
    text = path.read_text(encoding="utf-8")
    assert text.count("LLM_MODEL=") == 1
    assert "# keep this comment" in text
    assert dotenv_values(path, interpolate=False)["OPENALEX_API_KEY"] == "unrelated"
    assert load_gui_configuration(path) == configuration
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_failed_atomic_replace_preserves_previous_settings(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("LLM_PROVIDER=placeholder\nKEEP=untouched\n", encoding="utf-8")
    monkeypatch.setattr(gui_settings.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError):
        save_gui_configuration(LLMConfiguration("openai-compatible", "model", "secret"), path)
    assert path.read_text(encoding="utf-8") == "LLM_PROVIDER=placeholder\nKEEP=untouched\n"
    assert list(tmp_path.glob(".env.*")) == []


def test_offline_save_removes_persisted_key(tmp_path):
    path = tmp_path / ".env"
    current = LLMConfiguration("openai-compatible", "some-model", "temporary-secret")
    save_gui_configuration(current, path)
    save_gui_configuration(configure_provider("offline", "", current), path)
    assert load_gui_configuration(path).provider == "placeholder"
    assert dotenv_values(path, interpolate=False)["LLM_API_KEY"] == ""
    assert "temporary-secret" not in path.read_text(encoding="utf-8")


def test_dotenv_disabled_prevents_default_file_from_loading(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    save_gui_configuration(LLMConfiguration("openai-compatible", "model", "test-secret"), path)
    monkeypatch.setattr(gui_settings, "DEFAULT_ENV_PATH", path)
    monkeypatch.setattr(gui_settings, "ENV_PATH", path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "placeholder")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert load_gui_configuration().provider == "placeholder"
