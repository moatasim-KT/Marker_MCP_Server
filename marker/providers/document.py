"""Copyright (c) 2024 Marker Contributors. All rights reserved.

This module provides the DocumentProvider class for converting DOCX files to PDF,
including handling of embedded images and custom styling. It extends PdfProvider
and uses Mammoth and WeasyPrint for conversion.
"""

import base64
import re
import tempfile
from io import BytesIO
from pathlib import Path

import mammoth
from PIL import Image
from pydantic import BaseModel
from weasyprint import CSS, HTML

from marker.logger import get_logger
from marker.providers.pdf import PdfProvider

logger = get_logger()

css = """
@page {
    size: A4;
    margin: 2cm;
}

img {
    max-width: 100%;
    max-height: 25cm;
    object-fit: contain;
    margin: 12pt auto;
}

div, p {
    max-width: 100%;
    word-break: break-word;
    font-size: 10pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    break-inside: auto;
    font-size: 10pt;
}

tr {
    break-inside: avoid;
    page-break-inside: avoid;
}

td {
    border: 0.75pt solid #000;
    padding: 6pt;
}
"""


class DocumentProvider(PdfProvider):
    """Provides functionality to convert DOCX files to PDF, handling embedded images and custom styling.

    Inherits from PdfProvider.
    """

    def __init__(self, filepath: str, config: BaseModel | dict | None = None) -> None:
        """Initialize the DocumentProvider by converting a DOCX file to a temporary PDF file.

        Args:
            filepath (str): Path to the DOCX file to convert.
            config (optional): Configuration for the PDF provider.

        Raises:
            RuntimeError: If DOCX to PDF conversion fails.

        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            self.temp_pdf_path = temp_pdf.name

        # Convert DOCX to PDF
        try:
            self.convert_docx_to_pdf(filepath)
        except Exception as err:
            msg = f"Failed to convert {filepath} to PDF: {err}"
            raise RuntimeError(msg) from err

        # Initialize the PDF provider with the temp pdf path
        super().__init__(self.temp_pdf_path, config)

    def __del__(self) -> None:
        """Clean up the temporary PDF file created during initialization."""
        temp_pdf_path = Path(self.temp_pdf_path)
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()

    def convert_docx_to_pdf(self, filepath: str) -> None:
        """Convert a DOCX file to PDF using Mammoth for HTML conversion and WeasyPrint for PDF rendering.

        Args:
            filepath (str): Path to the DOCX file to convert.

        """
        docx_path = Path(filepath)
        with docx_path.open("rb") as docx_file:
            # we convert the docx to HTML
            result = mammoth.convert_to_html(docx_file)
            html = result.value

            # We convert the HTML into a PDF
            HTML(string=self._preprocess_base64_images(html)).write_pdf(
                self.temp_pdf_path,
                stylesheets=[CSS(string=css), self.get_font_css()],
            )

    @staticmethod
    def _preprocess_base64_images(html_content: str) -> str:
        pattern = r'data:([^;]+);base64,([^"\'>\s]+)'

        def convert_image(match: re.Match) -> str:
            try:
                img_data = base64.b64decode(match.group(2))
                with BytesIO(img_data) as bio, Image.open(bio) as img:
                    output = BytesIO()
                    img.save(output, format=img.format)
                    new_base64 = base64.b64encode(output.getvalue()).decode()
                    return f"data:{match.group(1)};base64,{new_base64}"
            except Exception:
                logger.exception("Failed to process image")
                return ""  # we ditch broken images as that breaks the PDF creation down the line

        return re.sub(pattern, convert_image, html_content)
