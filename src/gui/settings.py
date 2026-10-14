"""Session-local GUI settings; never write credentials or mutate the environment."""

from __future__ import annotations

import urllib.parse
from dataclasses import replace

import streamlit as st

from src.llm import LLMConfiguration


ROLE_LABELS = {
    "research_step": "Choose next activity",
    "proof_attempt": "Proof attempts",
    "literature_review": "Literature review",
    "analysis": "Analysis",
    "partial_result": "Partial results",
    "conjecture": "Conjectures",
    "open_problem": "Open problems",
    "bounded_experiment": "Experiment planning",
    "literature_selection": "Literature source selection",
    "literature_extraction": "Literature claim extraction",
    "unit_selection": "Choose research unit",
}

# Convenience defaults, not a model catalogue or a promise of free API access.
PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "gpt-4.1-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
}


def validate_endpoint(url: str) -> str:
    url = url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTPS API base URL without credentials, query parameters or fragments.")
    return url


def provider_label(configuration: LLMConfiguration) -> str:
    if configuration.provider == "placeholder":
        return "offline"
    for name, (endpoint, _) in PROVIDER_DEFAULTS.items():
        if configuration.base_url.rstrip("/") == endpoint:
            return name
    return "custom"


def configure_provider(name: str, key: str, current: LLMConfiguration) -> LLMConfiguration:
    """Apply an explicit provider choice without reusing credentials across endpoints."""
    name = name.strip().casefold()
    name = {"google": "gemini", "google gemini": "gemini", "placeholder": "offline",
            "openai-compatible": "custom"}.get(name, name)
    if name == "offline":
        return replace(current, provider="placeholder", api_key="")
    if name == "custom":
        endpoint, model = validate_endpoint(current.base_url), current.model
    elif name in PROVIDER_DEFAULTS:
        endpoint, model = PROVIDER_DEFAULTS[name]
    else:
        raise ValueError("Enter OpenAI, Gemini, offline, or custom (set its endpoint under Advanced).")
    same_endpoint = endpoint == current.base_url.rstrip("/")
    if same_endpoint and current.model and current.model != "placeholder-model" and current.provider != "placeholder":
        model = current.model
    api_key = key.strip() or (current.api_key if same_endpoint else "")
    if not api_key:
        raise ValueError("Enter a key for this provider; keys are never carried to a different endpoint.")
    return LLMConfiguration("openai-compatible", model, api_key, endpoint)


def settings_view() -> None:
    st.header("Settings")
    st.caption("Session only. Nothing is written to .env or the database. Use this interface locally.")
    configuration = st.session_state.get("api_configuration") or LLMConfiguration.from_environment()
    if st.session_state.pop("settings_clear_models", False):
        for role in ["default", *ROLE_LABELS]:
            st.session_state.pop(f"settings_model_{role}", None)
    if message := st.session_state.pop("settings_notice", None):
        st.success(message)
    with st.form("connection_settings"):
        provider = st.text_input("Provider name", value=provider_label(configuration), key="settings_provider",
                                 help="Type OpenAI or Gemini. Use offline to disable API calls, or custom for another compatible endpoint.")
        key = st.text_input("API key", type="password", key="settings_api_key",
                            help="Leave blank to keep the current key for the same endpoint.")
        applied = st.form_submit_button("Save API settings")
    if applied:
        try:
            updated = configure_provider(provider, key, configuration)
            if updated.base_url != configuration.base_url or updated.provider == "placeholder":
                st.session_state["model_routes"] = {}
            st.session_state["api_configuration"] = updated
            st.session_state["settings_clear_models"] = True
            st.session_state.pop("settings_api_url", None)
            st.session_state["clear_api_key_field"] = True
            st.session_state["settings_notice"] = "API settings saved for this session. No API request was made."
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.caption(f"Active model: {configuration.model} · "
               + ("API key configured" if configuration.remote_enabled else "Offline / no active key"))

    with st.expander("Advanced: model IDs, endpoint and activity overrides"):
        st.caption("Optional. All models use this one endpoint/key. Blank activity fields inherit the default "
                   "or their activity family. Model names do not enable web search or a deep-research service.")
        with st.form("model_settings"):
            endpoint = st.text_input("API base URL", value=configuration.base_url, key="settings_api_url")
            default = st.text_input("Default model ID", value=configuration.model, key="settings_model_default")
            routes = st.session_state.get("model_routes", {})
            selected = {
                role: st.text_input(label, value=routes.get(role, ""), key=f"settings_model_{role}", placeholder="Inherit")
                for role, label in ROLE_LABELS.items()
            }
            if st.form_submit_button("Apply advanced settings"):
                try:
                    endpoint = validate_endpoint(endpoint)
                    if not default.strip():
                        raise ValueError("Enter a default model ID.")
                    changed = endpoint != configuration.base_url.rstrip("/")
                    st.session_state["api_configuration"] = replace(
                        configuration, base_url=endpoint, model=default.strip(),
                        api_key="" if changed else configuration.api_key,
                        provider="openai-compatible" if changed else configuration.provider,
                    )
                    st.session_state["model_routes"] = {} if changed else {
                        role: value.strip() for role, value in selected.items() if value.strip()
                    }
                    st.session_state.pop("settings_provider", None)
                    st.session_state["clear_api_key_field"] = True
                    if changed:
                        st.session_state["settings_notice"] = "Endpoint changed; old key and overrides cleared. Enter the new endpoint's API key above."
                        st.session_state["settings_clear_models"] = True
                    else:
                        st.session_state["settings_notice"] = "Advanced settings saved for subsequent runs."
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with st.expander("Database connection"):
        with st.form("database_settings"):
            database_url = st.text_input("Database URL", value=st.session_state["database_url"],
                                         type="password", key="settings_database_url")
            if st.form_submit_button("Apply database"):
                if not database_url.strip():
                    st.error("Enter a database URL.")
                else:
                    st.session_state["database_url"] = database_url.strip()
                    st.success("Database connection updated for this session.")

    if st.button("Reset session settings", key="reset_settings"):
        for name in list(st.session_state):
            if name.startswith("settings_") or name in {
                "api_configuration", "model_routes", "model_catalog", "database_url", "clear_api_key_field",
            }:
                del st.session_state[name]
        st.rerun()
