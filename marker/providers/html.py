"""HTML to PDF provider module for converting HTML files to PDF using WeasyPrint."""
import tempfile
from pathlib import Path
from typing import Optional

from weasyprint import HTML
from pydantic import BaseModel

from marker.providers.pdf import PdfProvider


class HTMLProvider(PdfProvider):
    """Provider for converting HTML files to PDF and handling them as PDFs."""

    def __init__(self, filepath: str, config: Optional[BaseModel | dict] = None) -> None:
        """Initialize the HTMLProvider by converting HTML to a temporary PDF file.

        Args:
            filepath (str): Path to the HTML file to convert.
            config (dict | None, optional): Configuration for the PDF provider. Defaults to None.

        Raises:
            RuntimeError: If the HTML file cannot be converted to PDF.

        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            self.temp_pdf_path = temp_pdf.name

        # Convert HTML to PDF
        try:
            self.convert_html_to_pdf(filepath)
        except OSError as e:
            msg = f"Failed to convert {filepath} to PDF: {e}"
            raise RuntimeError(msg) from e

        # Initialize the PDF provider with the temp pdf path
        super().__init__(self.temp_pdf_path, config)

    def __del__(self) -> None:
        """Clean up the temporary PDF file on deletion."""
        if hasattr(self, "temp_pdf_path") and Path(self.temp_pdf_path).exists():
            Path(self.temp_pdf_path).unlink()

    def convert_html_to_pdf(self, filepath: str) -> None:
        """Convert an HTML file to PDF using WeasyPrint.

        Args:
            filepath (str): Path to the HTML file to convert.

        """
        font_css = self.get_font_css()
        HTML(filename=filepath, encoding="utf-8").write_pdf(
            self.temp_pdf_path, stylesheets=[font_css],
        )
