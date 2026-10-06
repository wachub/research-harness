"""Session-local GUI settings; never write credentials or mutate the environment."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
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


def validate_endpoint(url: str) -> str:
    url = url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTPS API base URL without credentials, query parameters or fragments.")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Authorization header to a redirect destination.
        return None


def load_model_ids(configuration: LLMConfiguration) -> list[str]:
    """Explicit metadata request only: no completion or research is performed."""
    endpoint = validate_endpoint(configuration.base_url)
    if not configuration.remote_enabled:
        raise ValueError("Apply an API connection and key first.")
    request = urllib.request.Request(
        endpoint + "/models", headers={"Authorization": f"Bearer {configuration.api_key}"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=15) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Model list exceeds the size limit.")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("The endpoint did not return an OpenAI-compatible model list.")
        models = sorted({item["id"] for item in payload["data"]
                         if isinstance(item, dict) and isinstance(item.get("id"), str)
                         and 0 < len(item["id"]) <= 200})
        if not models:
            raise ValueError("No models were returned. You can enter a model ID manually.")
        return models
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Model list request failed (HTTP {exc.code}). Check the endpoint and key.") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Could not read the model list. You can still enter a model ID manually.") from None


def settings_view() -> None:
    st.header("Settings")
    st.caption("Session only: settings survive page changes, but not a new browser session or server restart. "
               "Nothing is written to .env or the database. Use this credential-entry interface locally.")
    configuration = st.session_state.get("api_configuration") or LLMConfiguration.from_environment()
    with st.form("connection_settings"):
        st.subheader("API connection")
        mode = st.selectbox("Connection", ["OpenAI-compatible API", "Offline"],
                            index=0 if configuration.remote_enabled else 1, key="settings_api_mode")
        endpoint = st.text_input("API base URL", value=configuration.base_url, key="settings_api_url",
                                 help="Use your provider's OpenAI-compatible base URL, not a chat website.")
        key = st.text_input("API key", type="password", key="settings_api_key",
                            help="Leave blank to retain the current key only when the endpoint is unchanged.")
        st.caption("A key is configured." if configuration.api_key else "No key configured.")
        applied = st.form_submit_button("Apply connection")
    if applied:
        try:
            if mode not in {"OpenAI-compatible API", "Offline"}:
                raise ValueError("Unknown connection type.")
            endpoint = validate_endpoint(endpoint)
            changed = endpoint != configuration.base_url.rstrip("/")
            api_key = key.strip() or (configuration.api_key if not changed else "")
            if mode != "Offline" and not api_key:
                raise ValueError("Enter a key for this API endpoint.")
            configuration = replace(configuration, provider="placeholder" if mode == "Offline" else "openai-compatible",
                                    api_key="" if mode == "Offline" else api_key, base_url=endpoint)
            st.session_state["api_configuration"] = configuration
            if changed:
                st.session_state["model_catalog"] = []
                st.session_state["model_routes"] = {}
                for role in ["default", *ROLE_LABELS]:
                    st.session_state.pop(f"settings_model_{role}", None)
            # Remove the submitted key field on the next render, not after widget creation.
            st.session_state["clear_api_key_field"] = True
            st.session_state["settings_notice"] = "Connection applied. No API request was made."
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if message := st.session_state.pop("settings_notice", None):
        st.success(message)
    st.subheader("Models by activity")
    st.caption("Load model IDs from the active API, then select them below. Availability does not imply free usage "
               "or structured-response support. Changing a model does not enable a deep-research service.")
    if st.button("Load models from API", disabled=not configuration.remote_enabled, key="load_api_models"):
        try:
            with st.spinner("Loading model names…"):
                st.session_state["model_catalog"] = load_model_ids(configuration)
            st.success("Model list loaded; no research or completion was requested.")
        except ValueError as exc:
            st.warning(str(exc))
    routes = st.session_state.get("model_routes", {})
    catalog = sorted(set(st.session_state.get("model_catalog", []))
                     | {configuration.model} | set(routes.values()))
    with st.form("model_settings"):
        default = st.selectbox("Default model", catalog, index=catalog.index(configuration.model),
                               accept_new_options=True, key="settings_model_default")
        st.caption("Default applies unless overridden. Literature substeps inherit the literature model; "
                   "other activities inherit the activity-selection model.")
        selected = {}
        left, right = st.columns(2)
        for index, (role, label) in enumerate(ROLE_LABELS.items()):
            options = [None, *catalog]
            with left if index % 2 == 0 else right:
                selected[role] = st.selectbox(
                    label, options, index=options.index(routes.get(role)),
                    format_func=lambda value: "Inherit" if value is None else value,
                    accept_new_options=True, key=f"settings_model_{role}",
                )
        if st.form_submit_button("Apply models"):
            if not default or not default.strip():
                st.error("Choose a default model.")
            else:
                st.session_state["api_configuration"] = replace(configuration, model=default.strip())
                st.session_state["model_routes"] = {role: value.strip() for role, value in selected.items()
                                                   if value and value.strip()}
                st.success("Models applied to subsequent runs in this session.")

    with st.form("database_settings"):
        st.subheader("Database")
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
