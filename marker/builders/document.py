from typing import Annotated, Type

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """
    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for Layout and Line Detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False

    def __call__(self, provider: PdfProvider, layout_builder: LayoutBuilder, line_builder: LineBuilder, ocr_builder: OcrBuilder):
        document = self.build_document(provider)
        layout_builder(document, provider)
        line_builder(document, provider)
        if not self.disable_ocr:
            ocr_builder(document, provider)
        return document

    def build_document(self, provider: PdfProvider):
        page_range = provider.page_range if provider.page_range is not None else []
        lowres_images = provider.get_images(page_range, self.lowres_image_dpi)
        highres_images = provider.get_images(page_range, self.highres_image_dpi)
        initial_pages = []
        for i, p in enumerate(page_range):
            polygon = provider.get_page_bbox(p)
            if polygon is None:
                polygon = PolygonBox(polygon=[])
            refs = provider.get_page_refs(p)
            initial_pages.append(
                PageGroup(
                    page_id=p,
                    lowres_image=lowres_images[i],
                    highres_image=highres_images[i],
                    polygon=polygon,
                    refs=refs
                )
            )
        return Document(filepath=provider.filepath, pages=initial_pages)
