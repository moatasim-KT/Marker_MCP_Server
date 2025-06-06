"""EpubProvider module for converting EPUB files to PDF and providing PDF-based document processing.

This module defines the EpubProvider class, which extends PdfProvider to support EPUB file conversion and processing using ebooklib, BeautifulSoup, and WeasyPrint.
"""

import base64
import tempfile
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub
from weasyprint import CSS, HTML

from marker.providers.pdf import PdfProvider

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


class EpubProvider(PdfProvider):
    """Provider for converting EPUB files to PDF and providing PDF-based document processing."""

    def __init__(self, filepath: str, config: dict | None = None) -> None:
        """Initialize EpubProvider by converting EPUB to PDF and initializing the PDF provider.

        Args:
            filepath (str): Path to the EPUB file.
            config (Optional[dict], optional): Configuration dictionary. Defaults to None.

        Raises:
            RuntimeError: If EPUB to PDF conversion fails.

        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            self.temp_pdf_path = temp_pdf.name

        # Convert Epub to PDF
        try:
            self.convert_epub_to_pdf(filepath)
        except Exception as e:
            msg = f"Failed to convert {filepath} to PDF: {e}"
            raise RuntimeError(msg) from e

        # Initialize the PDF provider with the temp pdf path
        super().__init__(self.temp_pdf_path, config)

    def __del__(self) -> None:
        """Clean up the temporary PDF file on deletion."""
        if hasattr(self, "temp_pdf_path") and Path(self.temp_pdf_path).exists():
            Path(self.temp_pdf_path).unlink()

    @staticmethod
    def _extract_images_and_styles(ebook: epub.EpubBook) -> tuple[dict[str, str], list[str]]:
        """Extract images and styles from the EPUB ebook.

        Args:
            ebook: The EPUB ebook object.

        Returns:
            tuple[dict[str, str], list[str]]: A tuple containing a dictionary of image tags and a list of styles.

        """
        img_tags = {}
        styles = []
        for item in ebook.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                img_data = base64.b64encode(item.get_content()).decode("utf-8")
                img_tags[item.file_name] = f"data:{item.media_type};base64,{img_data}"
            elif item.get_type() == ebooklib.ITEM_STYLE:
                styles.append(item.get_content().decode("utf-8"))
        return img_tags, styles

    @staticmethod
    def _extract_html_content(ebook: epub.EpubBook) -> str:
        """Extract HTML content from the EPUB ebook.

        Args:
            ebook: The EPUB ebook object.

        Returns:
            str: The concatenated HTML content from the EPUB.

        """
        html_content = ""
        for item in ebook.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                html_content += item.get_content().decode("utf-8")
        return html_content

    @staticmethod
    def _replace_image_tags(soup: BeautifulSoup, img_tags: dict[str, str]) -> None:
        """Replace image sources in the soup with base64 data.

        Args:
            soup (BeautifulSoup): The parsed HTML soup.
            img_tags (dict[str, str]): Mapping of image filenames to base64 data URLs.

        """
        for img in soup.find_all("img"):
            if isinstance(img, Tag):
                src = img.get("src")
                if isinstance(src, str) and src:
                    normalized_src = src.replace("../", "")
                    if normalized_src in img_tags:
                        img["src"] = img_tags[normalized_src]
        for image in soup.find_all("image"):
            if isinstance(image, Tag):
                src = image.get("xlink:href")
                if isinstance(src, str) and src:
                    normalized_src = src.replace("../", "")
                    if normalized_src in img_tags:
                        image["xlink:href"] = img_tags[normalized_src]

    def convert_epub_to_pdf(self, filepath: str) -> None:
        """Convert an EPUB file to PDF using ebooklib and WeasyPrint.

        Args:
            filepath (str): Path to the EPUB file.

        """
        ebook = epub.read_epub(filepath)
        img_tags, _ = self._extract_images_and_styles(ebook)
        html_content = self._extract_html_content(ebook)
        soup = BeautifulSoup(html_content, "html.parser")
        self._replace_image_tags(soup, img_tags)
        html_content = str(soup)
        full_style = f"{css}"
        # we convert the epub to HTML
        HTML(string=html_content, base_url=filepath).write_pdf(
            self.temp_pdf_path,
            stylesheets=[CSS(string=full_style), self.get_font_css()],
        )
