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
    scheme = parsed.scheme or ("https" if host.endswith(".qdrant.io") else "http")

    # Always coerce Qdrant traffic through the canonical HTTP port.  Qdrant Cloud
    # listeners forward 443 → 6333, so we normalise every endpoint to include the
    # explicit port to avoid ``/aliases`` 404s from hitting the management plane.
    port = 6333
    if ":" in host and not host.startswith("["):
        host_fmt = f"[{host}]"
    else:
        host_fmt = host
    netloc = f"{host_fmt}:{port}"
    return urlunparse((scheme, netloc, "", "", "", ""))


__all__ = ["normalize_qdrant_url"]
