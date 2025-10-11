"""Network-related utility helpers."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_qdrant_url(url: str | None) -> str:
    """Return a normalized Qdrant base URL with a guaranteed scheme and port.

    Qdrant Cloud endpoints expect HTTPS and typically omit an explicit port
    while local deployments usually live on ``http://localhost:6333``.  This
    helper ensures a consistent ``scheme://host:port`` layout so downstream
    clients hit ``:6333`` regardless of the user-provided input.
    """

    if not url:
        return "http://localhost:6333"

    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"

    parsed = urlparse(candidate)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6333
    if parsed.scheme:
        scheme = parsed.scheme
    else:
        scheme = "https" if host.endswith(".qdrant.io") else "http"

    netloc = f"{host}:{port}"
    return urlunparse((scheme, netloc, "", "", "", ""))


__all__ = ["normalize_qdrant_url"]
