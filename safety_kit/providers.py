"""Provider registry — maps provider name to API base URL and env var.

Supported providers:
    openai   → OpenAI API        (OPENAI_API_KEY)
    groq     → Groq API          (GROQ_API_KEY)
    gemini   → Google Gemini     (GEMINI_API_KEY)
    ollama   → Local Ollama      (no key needed)

Model string format:  "<provider>/<model-name>"
Examples:
    "openai/gpt-4o-mini"
    "groq/llama-3.1-8b-instant"
    "gemini/gemini-1.5-flash"
    "ollama/llama3"
    "gpt-4o-mini"          ← no prefix → treated as openai
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    base_url: str | None
    api_key_env: str


_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        base_url=None,  # default openai SDK endpoint
        api_key_env="OPENAI_API_KEY",
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="",  # no key required
    ),
}


def resolve(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Parse a model string and return (model_name, base_url, api_key).

    The caller can still pass explicit ``api_key`` / ``base_url`` overrides —
    those take precedence over auto-detected values.

    Args:
        model: Model identifier, optionally prefixed with the provider name.
        api_key: Explicit API key override (optional).
        base_url: Explicit base URL override (optional).

    Returns:
        (model_name, resolved_base_url, resolved_api_key)

    Raises:
        ValueError: If the provider prefix is not recognised.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
    else:
        provider = "openai"
        model_name = model

    provider = provider.lower()

    if provider not in _PROVIDERS:
        known = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{provider}'. Known providers: {known}. "
            f"Pass 'base_url' explicitly to use a custom endpoint."
        )

    cfg = _PROVIDERS[provider]

    resolved_base_url = base_url or cfg.base_url

    if api_key is None:
        env_var = cfg.api_key_env
        api_key = os.getenv(env_var) if env_var else None

    return model_name, resolved_base_url, api_key


def list_providers() -> list[str]:
    """Return the names of all registered providers."""
    return list(_PROVIDERS.keys())
