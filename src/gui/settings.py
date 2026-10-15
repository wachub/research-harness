"""Minimal, local API configuration for the Streamlit GUI."""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
from dataclasses import replace
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from src.llm import LLMConfiguration


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
ENV_PATH = DEFAULT_ENV_PATH
PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "gpt-4.1-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
}
LLM_ENV_KEYS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL")
_LLM_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?(LLM_PROVIDER|LLM_MODEL|LLM_API_KEY|LLM_BASE_URL)\s*=")


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


def load_gui_configuration(path: Path | None = None) -> LLMConfiguration:
    """Prefer the GUI's local saved settings over process-level defaults."""
    path = path or ENV_PATH
    if os.getenv("PYTHON_DOTENV_DISABLED") == "1" and path == DEFAULT_ENV_PATH:
        return LLMConfiguration.from_environment()
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    if all(key in values for key in LLM_ENV_KEYS):
        return LLMConfiguration(
            provider=(values["LLM_PROVIDER"] or "placeholder").strip().lower(),
            model=(values["LLM_MODEL"] or "placeholder-model").strip(),
            api_key=(values["LLM_API_KEY"] or "").strip(),
            base_url=(values["LLM_BASE_URL"] or "https://api.openai.com/v1").strip().rstrip("/"),
        )
    return LLMConfiguration.from_environment()


def configure_provider(
    name: str, key: str, current: LLMConfiguration, *, model: str = "", base_url: str = "",
) -> LLMConfiguration:
    """Resolve a typed provider and never carry a key to a different endpoint."""
    name = name.strip().casefold()
    name = {"google": "gemini", "google gemini": "gemini", "placeholder": "offline",
            "openai-compatible": "custom"}.get(name, name)
    if name == "offline":
        return replace(current, provider="placeholder", model="placeholder-model", api_key="")
    if name == "custom":
        endpoint = validate_endpoint(base_url or (current.base_url if provider_label(current) == "custom" else ""))
        default_model = ""
    elif name in PROVIDER_DEFAULTS:
        endpoint, default_model = PROVIDER_DEFAULTS[name]
        if base_url.strip() and validate_endpoint(base_url) != endpoint:
            raise ValueError("Use provider 'custom' for a different API base URL.")
    else:
        raise ValueError("Enter OpenAI, Gemini, custom, or offline.")
    same_endpoint = endpoint == current.base_url.rstrip("/") and current.provider != "placeholder"
    chosen_model = model.strip() or (current.model if same_endpoint else default_model)
    if not re.fullmatch(r"[A-Za-z0-9._:/-]{1,160}", chosen_model):
        raise ValueError("Enter a valid model ID for this provider.")
    api_key = key.strip() or (current.api_key if same_endpoint else "")
    if not api_key:
        raise ValueError("Enter an API key for this provider; keys are not reused across endpoints.")
    if any(ord(character) < 32 for character in api_key):
        raise ValueError("The API key must be a single line.")
    return LLMConfiguration("openai-compatible", chosen_model, api_key, endpoint)


def save_gui_configuration(configuration: LLMConfiguration, path: Path | None = None) -> None:
    """Atomically update only LLM settings in the git-ignored local .env file."""
    path = path or ENV_PATH
    if path.is_symlink():
        raise OSError("The settings file must not be a symbolic link.")
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    retained = "".join(line for line in previous.splitlines(keepends=True)
                       if not _LLM_ASSIGNMENT.match(line))
    values = {
        "LLM_PROVIDER": configuration.provider,
        "LLM_MODEL": configuration.model,
        "LLM_API_KEY": configuration.api_key,
        "LLM_BASE_URL": configuration.base_url,
    }
    content = retained.rstrip("\n") + "\n" if retained else ""
    content += "".join(f"{key}={json.dumps(values[key])}\n" for key in LLM_ENV_KEYS)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".env.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def settings_view() -> None:
    st.header("API connection")
    st.caption("Settings are saved in this computer's git-ignored .env file. Use this page only on a local, private server.")
    configuration = st.session_state["api_configuration"]
    if message := st.session_state.pop("settings_notice", None):
        st.success(message)
    with st.container(border=True):
        label = provider_label(configuration)
        state = "Key configured" if configuration.remote_enabled else "Offline"
        st.caption(f"Current: {label} · {configuration.model} · {state}")
        with st.form("api_settings"):
            provider = st.text_input("Provider name", value=label, key="settings_provider",
                                     help="OpenAI, Gemini, custom, or offline.")
            model = st.text_input("Model ID", key="settings_model",
                                  help="Optional for OpenAI and Gemini. Blank keeps the current model for the same provider.")
            key = st.text_input("API key", type="password", key="settings_api_key",
                                help="Leave blank to keep the current key for the same endpoint. The existing key is never shown.")
            with st.expander("Custom endpoint"):
                base_url = st.text_input("API base URL", key="settings_api_url",
                                         help="Required only when provider name is custom. Use the HTTPS base URL, not /chat/completions.")
            submitted = st.form_submit_button("Save API connection", type="primary")
    if submitted:
        try:
            updated = configure_provider(provider, key, configuration, model=model, base_url=base_url)
            save_gui_configuration(updated)
        except ValueError as exc:
            st.error(str(exc))
        except OSError:
            st.error("Could not save the local .env file. Check its permissions and try again.")
        else:
            st.session_state["api_configuration"] = updated
            st.session_state["model_routes"] = {}
            st.session_state["settings_clear_fields"] = True
            st.session_state["settings_notice"] = "API settings saved locally. No API request was made."
            st.rerun()
