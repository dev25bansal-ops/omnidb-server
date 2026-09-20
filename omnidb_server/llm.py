from __future__ import annotations

import httpx

_OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434/v1"
_PROBE_TIMEOUT_S = 1.5


def probe(base_url: str) -> list[str] | None:
    """Model ids from an OpenAI-compatible {base}/models endpoint.

    Returns None on any connection/parse failure or when no models are
    listed. Never raises.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=_PROBE_TIMEOUT_S)
        response.raise_for_status()
        data = response.json().get("data") or []
        ids = [entry["id"] for entry in data if isinstance(entry, dict) and entry.get("id")]
        return ids or None
    except Exception:
        return None


def resolve_llm_config(cfg) -> tuple[str | None, str | None]:
    """Resolve (base_url, model) for consolidation.

    Explicit config wins. With ollama_autodetect on, probe the configured
    endpoint (or the Ollama default) once - 1.5s budget - and take the first
    listed model unless one is pinned. Any failure yields (None, None) so
    consolidation runs the offline stub instead. Never raises: autodetect is
    best-effort by design.
    """
    base_url, model = cfg.llm_base_url, cfg.llm_model
    if (base_url and model) or not cfg.ollama_autodetect:
        return base_url, model
    endpoint = base_url or _OLLAMA_DEFAULT_BASE
    ids = probe(endpoint)
    if ids is None:
        return None, None
    return endpoint, model or ids[0]
