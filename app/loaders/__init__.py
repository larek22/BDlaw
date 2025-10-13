from .pdf_loader import load_pdf
from .docx_loader import load_docx
from .rtf_loader import load_rtf
from .html_loader import load_html
from .txt_loader import load_txt

__all__ = [
    "load_pdf",
    "load_docx",
    "load_rtf",
    "load_html",
    "load_txt",
]
