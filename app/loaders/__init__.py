"""Unified document loaders used by the atomic ingestion pipeline."""

from .docx_loader import load_docx  # noqa: F401
from .html_loader import load_html  # noqa: F401
from .pdf_loader import load_pdf  # noqa: F401
from .rtf_loader import load_rtf  # noqa: F401
from .txt_loader import load_txt  # noqa: F401

__all__ = [
    "load_docx",
    "load_html",
    "load_pdf",
    "load_rtf",
    "load_txt",
]
